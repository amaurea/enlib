"""This module contains classes for representing time-ordered data from a
scanning instrument such as a radio telescope, which consists of a set of
detectors with a constant relative orientation, with the set of detectors
moving according to a "boresight", and with each detector measuring a
certain linear combination of some variables (usually stokes parameters).
The actual detector samples are not directly exposed, as they will often
be too expensive to store. Instead, a get_samples() function is provided,
which can be overriden in subclasses.

Scans shall be sliceable. The first index (detectors) takes an arbitrary
slice, including arrays of indices etc. The second index can only be
a plain slice.


How to interpret the step property of that slice?

 1. It is the  level of decimation. So scan[:,::2] will result in a
    new scan with data decimated by two. This means that
    scan[:,::2].get_samples() != san.get_samples()[:,::2].

 2. It has the normal meaning of skipping samples. One will then
    need to provide an additional "downgrade()" method. But it is
    less misleading, and allows you to get for example the odd
    or even samples if that's what you really want.

Since the tod probably isn't stored directly in the Scan, but
computed on the fly when you call get_samples(), repeated
slicing and downgrading can be cumbersome. If only skipping or
only downgrading is allowed, then one can combine any number of
slices into one effective slice. If they should be mixed, then
one may have to store a list of all the operations, and sequentially
apply then when get_samples() is called.

I prefer the simplicity of letting ::d mean downgrading. Letting it
skip samples seems like a way to sneak in cuts by the back door, and
we already have a way to implement cuts (though not as efficient).
A mix of skipped and downgraded samples would make it much more difficult
to implement subclasses. For example - one wouldn't be able to look at the
values in boresight to determine the effective duration of each sample.
I therefore go with option #1 above, and explicitly do not provide a way
of skipping samples here.
"""

# This module has become really crusty. It's time to redo this whole
# system, especially considering the large overlap with Dataset.

from __future__ import division, print_function
import numpy as np, copy as cpy, h5py, os
from . import sampcut, nmat, config, resample, utils, bunch, fft

class Scan:
	"""This defines the minimal interface for a Scan. It will usually be
	inherited from."""
	def __init__(self, boresight=None, offsets=None, comps=None, tod=None, sys=None, cut=None, site=None, mjd0=0, noise=None, dets=None, id=""):
		# Boresight will always be unwound, i.e. it will have no 2pi jumps in it.
		# Time is measured in seconds since start of scan, with mjd0 indicating the MJD of the scan start.
		self.boresight = np.asfarray(boresight) # [nsamp,coords]
		self.offsets   = np.asfarray(offsets)   # [ndet,coords]
		self.comps     = np.asfarray(comps)     # [ndet,comps]
		self.cut       = cut.copy() if cut is not None else None # Sampcut
		self.noise     = noise
		self.id        = id # Identifier of this scan, for printing purposes
		# These are needed in order to interpret the coordinates
		self.sys       = str(sys)               # str
		self.site      = site
		self.mjd0      = mjd0                   # time basis
		self.dets      = np.arange(len(self.comps)) if dets is None else dets
		self.dgrid     = (1,np.max(self.dets)+1)
		self.hwp       = None
		self.mapping   = None
		# Not part of the general interface
		self._tod      = np.asfarray(tod)       # [ndet,nsamp]
	def get_samples(self, verbose=False):
		return self._tod.copy()
	@property
	def nsamp(self): return self.boresight.shape[0]
	@property
	def ndet(self): return self.comps.shape[0]
	def __repr__(self):
		return self.__class__.__name__ + "[ndet=%d,nsamp=%d]" % (self.ndet,self.nsamp)
	@property
	def box(self):
		return np.array([np.min(self.boresight,0)+np.min(self.offsets,0),np.max(self.boresight,0)+np.max(self.offsets,0)])
	@property
	def srate(self):
		step = self.boresight.shape[0]//100
		return float(step)/utils.medmean(self.boresight[::step,0][1:]-self.boresight[::step,0][:-1])
	@property
	def hwp_active(self): return self.hwp is not None and np.any(self.hwp_phase[0] != 0)
	def copy(self, shallow=False):
		if shallow: return cpy.copy(self)
		else:       return cpy.deepcopy(self)
	def getitem_helper(self, sel):
		if type(sel) != tuple: sel = (sel,)
		assert len(sel) < 3, "Too many indices in slice"
		detslice = sel[0] if len(sel) > 0 else slice(None)
		sampslice = sel[1] if len(sel) > 1 else slice(None)
		assert isinstance(sampslice,slice), "Sample part of slice must be slice object"
		res = cpy.deepcopy(self)
		# These will be passed to fortran, so make them contiguous
		res.boresight = np.ascontiguousarray(utils.slice_downgrade(res.boresight, sampslice, axis=0))
		res.offsets   = np.ascontiguousarray(res.offsets[detslice])
		res.comps     = np.ascontiguousarray(res.comps[detslice])
		res.dets      = res.dets[detslice]
		res.hwp       = np.ascontiguousarray(utils.slice_downgrade(res.hwp, sampslice, axis=0))
		res.hwp_phase = np.ascontiguousarray(utils.slice_downgrade(res.hwp_phase, sampslice, axis=0))
		try:
			# The whole scan stuff is horrible and should be redesigned
			res.dark_tod = np.ascontiguousarray(utils.slice_downgrade(res.dark_tod, sampslice, axis=1))
			res.dark_cut = res.dark_cut[sel]
		except AttributeError as e: pass
		try:
			res.buddy_comps = res.buddy_comps[:,detslice]
			res.buddy_offs  = res.buddy_offs[:,detslice]
		except AttributeError as e: pass
		res.cut       = res.cut[sel]
		res.cut_noiseest = res.cut_noiseest[sel]
		res.cut_basic = res.cut_basic[sel]
		res.noise     = res.noise[sel]
		return res, detslice, sampslice
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res._tod = np.ascontiguousarray(utils.slice_downgrade(res._tod[detslice], sampslice, axis=-1))
		return res
	def resample(self, mapping):
		res = cpy.deepcopy(self)
		res.boresight = np.ascontiguousarray(utils.interpol(res.boresight.T, mapping.oimap[None], order=1).T)
		res.hwp       = utils.interpol(utils.unwind(res.hwp), mapping.oimap[None], order=1)
		res.hwp_phase = np.ascontiguousarray(utils.interpol(res.hwp_phase.T, mapping.oimap[None], order=1).T)
		try:
			res.dark_tod = utils.interpol(res.dark_tod, mapping.oimap[None])
			res.dark_cut = resample_cut(res.dark_cut, mapping)
		except AttributeError as e: pass
		res.cut          = resample_cut(res.cut,          mapping)
		res.cut_noiseest = resample_cut(res.cut_noiseest, mapping)
		res.cut_basic    = resample_cut(res.cut_basic,    mapping)
		res.noise        = res.noise.resample(mapping)
		res.mapping      = mapping
		return res

config.default("downsample_method", "fft", "Method to use when downsampling the TOD")
class H5Scan(Scan):
	def __init__(self, fname):
		self.fname = fname
		with h5py.File(fname, "r") as hfile:
			for k in ["boresight","offsets","comps","sys","mjd0","dets"]:
				setattr(self, k, hfile[k].value)
			n = self.boresight.shape[0]
			ranges, detmap, nsamp = [hfile["cut/%s" % name].value for name in ["ranges","detmap","nsamp"]]
			self.cut  = sampcut.Sampcut(ranges, detmap, nsamp)
			self.cut_noiseest = self.cut.copy()
			self.noise= nmat.read_nmat(hfile, "noise")
			self.site = bunch.Bunch({k:hfile["site/"+k].value for k in hfile["site"]})
			self.subdets = np.arange(self.ndet)
			self.hwp = np.zeros(n)
			self.hwp_phase = np.zeros([n,2])
			self.sampslices = []
			self.id = os.path.basename(fname)
			self.entry = bunch.Bunch(id=self.id)
	def get_samples(self, verbose=False):
		"""Return the actual detector samples. Slow! Data is read from disk,
		so store the result if you need to reuse it."""
		with h5py.File(self.fname, "r") as hfile:
			tod = hfile["tod"].value[self.subdets]
		method = config.get("downsample_method")
		for s in self.sampslices:
			tod = resample.resample(tod, 1.0/np.abs(s.step or 1), method=method)
			s = slice(s.start, s.stop, np.sign(s.step) if s.step else None)
			tod = tod[:,s]
		res = np.ascontiguousarray(tod)
		return res
	def __repr__(self):
		return self.__class__.__name__ + "[ndet=%d,nsamp=%d,name=%s]" % (self.ndet,self.nsamp,self.fname)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res.sampslices.append(sampslice)
		res.subdets = res.subdets[detslice]
		return res

def write_scan(fname, scan):
	with h5py.File(fname, "w") as hfile:
		for k in ["boresight","offsets","comps","sys","mjd0","dets"]:
			hfile[k] = getattr(scan, k)
		ranges = scan.cut.ranges
		if ranges.size == 0: ranges = np.zeros([1,2],dtype=np.int32)
		hfile["cut/ranges"] = ranges
		hfile["cut/detmap"] = scan.cut.detmap
		hfile["cut/nsamp"]  = scan.cut.nsamp
		nmat.write_nmat(hfile.create_group("noise"), scan.noise)
		for k in scan.site:
			hfile["site/"+k] = scan.site[k]
		hfile["tod"]       = scan.get_samples()

def read_scan(fname):
	return H5Scan(fname)

# In the current (ugly) architecture, this is the least painful place to put
# HWP resampling.

def build_hwp_sample_mapping(hwp, quantile=0.1):
	"""Given a HWP angle, return an array with shape [nout] containing
	the original sample index (float) corresponding to each sample in the
	remapped array, along with the resulting hwp sample rate.
	The remapping also truncates the end to ensure that
	there is an integer number of HWP rotations in the data."""
	# Make sure there are no angle wraps in the hwp
	hwp = utils.unwind(hwp)
	# interp requires hwp to be monotonically increasing. In practice
	# it could be monotonically decreasing too, but our final result
	# does not depend on the direction of rotation, so we simply flip it here
	# if necessary
	hwp = np.abs(hwp)
	# Find the target hwp speed
	speed = np.percentile(hwp[1:]-hwp[:-1], 100*quantile)
	# We want a whole number of samples per revolution, and
	# a whole number of revolutions in the whole tod
	a    = hwp - hwp[0]
	nrev = int(np.floor(a[-1]/(2*np.pi)))
	nper = utils.nint(2*np.pi/speed)
	# Make each of these numbers fourier-friendly
	nrev = fft.fft_len(nrev, "below")
	nper = fft.fft_len(nper, "above")
	# Set up our output samples
	speed = 2*np.pi/nper
	nout  = nrev*nper
	ohwp  = hwp[0] + np.arange(nout)*speed
	# Find the input sample for each output sample
	res = bunch.Bunch()
	res.oimap = np.interp(ohwp, hwp, np.arange(len(hwp)))
	# Find the output sampe for each input sample too. Because of
	# cropping, the last of these will be invalid
	res.iomap = np.interp(np.arange(len(hwp)), res.oimap, np.arange(len(res.oimap)))
	# Find the average sampling rate change fsamp_rel = fsamp_out/fsamp_in
	res.fsamp_rel = 1/np.mean(res.oimap[1:]-res.oimap[:-1])
	res.insamp = len(hwp)
	res.onsamp = nout
	res.nrev   = nrev
	res.nper   = nper
	return res

def resample_cut(cut, mapping):
	"""Nearest neibhbor remapping of cut indices"""
	ocut = cut.copy()
	end = ocut.ranges>= mapping.insamp
	ocut.ranges[end] = mapping.onsamp
	ocut.ranges[~end] = mapping.iomap[cut.ranges[~end]].astype(int)
	ocut.nsamp  = mapping.onsamp
	# Widen cut because we had to round the cut indices
	ocut = ocut.widen(1)
	return ocut
