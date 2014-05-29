"""This module contains classes for representing time-ordered data from a
scanning instrument such as a radio telescope, which consists of a set of
detectors with a constant relative orientation, with the set of detectors
moving according to a "boresight", and with each detector measuring a
certain linear combination of some variables (usually stokes parameters).
The actual detector samples are not directly exposed, as they will often
be too expensive to store. Instead, a get_samples() function is provided,
which can be overriden in subclasses."""
import numpy as np

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
	def get_samples(self): return self._tod.copy()
	@property
	def nsamp(self): return self.boresight.shape[0]
	@property
	def ndet(self): return self.comps.shape[0]
	def __repr__(self):
		return self.__class__.__name__ + "[ndet=%d,nsamp=%d]" % (self.ndet,self.nsamp)
	@property
	def box(self):
		return np.array([np.min(self.boresight,0)+np.min(self.offsets,0),np.max(self.boresight,0)+np.max(self.offsets,0)])
