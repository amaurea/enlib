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
import numpy as np, enlib.utils, enlib.slice, copy as cpy

class Scan:
	"""This defines the minimal interface for a Scan. It will usually be
	inherited from."""
	def __init__(self, boresight=None, offsets=None, comps=None, tod=None, sys=None, cut=None, site=None, mjd0=0):
		# Boresight will always be unwound, i.e. it will have no 2pi jumps in it.
		# Time is measured in seconds since start of scan, with mjd0 indicating the MJD of the scan start.
		self.boresight = np.asfarray(boresight) # [nsamp,coords]
		self.offsets   = np.asfarray(offsets)   # [ndet,coords]
		self.comps     = np.asfarray(comps)     # [ndet,comps]
		self.cut       = rangelist.Multirange(cut) # Multirange[ndet,ranges]
		# These are needed in order to interpret the coordinates
		self.sys       = str(sys)               # str
		self.site      = site
		self.mjd0      = mjd0                   # time basis
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
		return float(step)/enlib.utils.medmean(self.boresight[::step,0][1:]-self.boresight[::step,0][:-1])
	def copy(self): return cpy.deepcopy(self)
	def getitem_helper(self, sel):
		detslice, sampslice = enlib.slice.split_slice(sel, (1,1))
		assert len(detslice) < 2 and len(sampslice) < 2, "Too many indices in slice"
		detslice  = slice(None) if len(detslice)  == 0 else detslice[0]
		sampslice = slice(None) if len(sampslice) == 0 else sampslice[0]
		assert isinstance(sampslice,slice), "Sample part of slice must be slice object"
		res = cpy.deepcopy(self)
		res.boresight = enlib.slice.slice_downgrade(res.boresight, sampslice, axis=0)
		res.offsets   = res.offsets[detslice]
		res.comps     = res.comps[detslice]
		res.cut       = res.cut[sel]
		res.dets      = res.dets[detslice]
		return res, detslice, sampslice
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res._tod = enlib.slice.slice_downgrade(res._tod[detslice], sampslice, axis=-1)
		return res
