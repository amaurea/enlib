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
import numpy as np, enlib.slice, copy as cpy, h5py, bunch
from enlib import rangelist, nmat, config, resample, utils

class Scan:
	"""This defines the minimal interface for a Scan. It will usually be
	inherited from."""
	def __init__(self, boresight=None, offsets=None, comps=None, tod=None, sys=None, cut=None, site=None, mjd0=0, noise=None, dets=None):
		# Boresight will always be unwound, i.e. it will have no 2pi jumps in it.
		# Time is measured in seconds since start of scan, with mjd0 indicating the MJD of the scan start.
		self.boresight = np.asfarray(boresight) # [nsamp,coords]
		self.offsets   = np.asfarray(offsets)   # [ndet,coords]
		self.comps     = np.asfarray(comps)     # [ndet,comps]
		self.cut       = rangelist.Multirange(cut) # Multirange[ndet,ranges]
		self.noise     = noise
		# These are needed in order to interpret the coordinates
		self.sys       = str(sys)               # str
		self.site      = site
		self.mjd0      = mjd0                   # time basis
		self.dets      = np.arange(len(self.comps)) if dets is None else dets
		self.dgrid     = (1,np.max(self.dets)+1)
		# Not part of the general interface
		self._tod      = np.asfarray(tod)       # [ndet,nsamp]
	def get_samples(self):
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
		step = self.boresight.shape[0]/100
		return float(step)/utils.medmean(self.boresight[::step,0][1:]-self.boresight[::step,0][:-1])
	def copy(self): return cpy.deepcopy(self)
	def getitem_helper(self, sel):
		if type(sel) != tuple: sel = (sel,)
		assert len(sel) < 3, "Too many indices in slice"
		detslice = sel[0] if len(sel) > 0 else slice(None)
		sampslice = sel[1] if len(sel) > 1 else slice(None)
		assert isinstance(sampslice,slice), "Sample part of slice must be slice object"
		res = cpy.deepcopy(self)
		# These will be passed to fortran, so make them contiguous
		res.boresight = np.ascontiguousarray(enlib.slice.slice_downgrade(res.boresight, sampslice, axis=0))
		res.offsets   = np.ascontiguousarray(res.offsets[detslice])
		res.comps     = np.ascontiguousarray(res.comps[detslice])
		res.dets      = res.dets[detslice]
		res.cut       = res.cut[sel]
		res.noise     = res.noise[sel]
		return res, detslice, sampslice
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res._tod = np.ascontiguousarray(enlib.slice.slice_downgrade(res._tod[detslice], sampslice, axis=-1))
		return res

config.default("downsample_method", "fft", "Method to use when downsampling the TOD")
class H5Scan(Scan):
	def __init__(self, fname):
		self.fname = fname
		with h5py.File(fname, "r") as hfile:
			for k in ["boresight","offsets","comps","sys","mjd0","dets"]:
				setattr(self, k, hfile[k].value)
			n = self.boresight.shape[0]
			neach = hfile["cut/neach"].value
			flat  = hfile["cut/flat"].value
			self.cut  = rangelist.Multirange((n,neach,flat),copy=False)
			self.noise= nmat.read_nmat(hfile, "noise")
			self.site = bunch.Bunch({k:hfile["site/"+k].value for k in hfile["site"]})
			self.subdets = np.arange(self.ndet)
			self.sampslices = []
	def get_samples(self):
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
		n, neach, flat = scan.cut.flatten()
		# h5py has problems with zero size arrays
		if flat.size == 0: flat = np.zeros([1,2],dtype=np.int32)
		hfile["cut/neach"] = neach
		hfile["cut/flat"]  = flat
		nmat.write_nmat(hfile, scan.noise, "noise")
		for k in scan.site:
			hfile["site/"+k] = scan.site[k]
		hfile["tod"]       = scan.get_samples()

def read_scan(fname):
	return H5Scan(fname)

default_site = bunch.Bunch(lat=0,lon=0,alt=0,T=273,P=550,hum=0.2,freq=100)
