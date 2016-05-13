"""This module provides classes for representing ranges of true and false values,
providing both a mask-like (numpy bool array) and list of from:to interface.
It also provides a convenience class for handling multiple of these range lists."""
import numpy as np
from enlib.slice import expand_slice, split_slice
from enlib.utils import mask2range, cumsum, range_union, range_normalize

class Rangelist:
	def __init__(self, ranges, n=None, copy=True):
		if isinstance(ranges, Rangelist):
			if copy: ranges = ranges.copy()
			self.n, self.ranges =  ranges.n, ranges.ranges
		else:
			ranges = np.asarray(ranges, dtype=int)
			if copy: ranges = np.array(ranges)
			if ranges.ndim == 1:
				self.n      = ranges.size
				self.ranges = mask2range(ranges)
			else:
				self.n      = int(n)
				# Since this class is supposed to be a sparese representation of a mask,
				# only non-overlapping non-empty ranges make sense.
				self.ranges = range_union(range_normalize(ranges))
	def __getitem__(self, sel):
		"""This function operates on the rangelist as if it were a dense numpy array.
		It returns either a sliced Rangelist or a bool."""
		if isinstance(sel,tuple):
			if len(sel) > 1: raise IndexError("Too many indices to Rangelist (only 1 supported)")
			if len(sel) == 0: return self
			sel = sel[0]
		if isinstance(sel,slice):
			sel = expand_slice(sel, self.n)
			if len(self.ranges) == 0:
				# Can't just return self here, as I did, because .n needs to be updated even if
				# ranges is empty.
				return Rangelist(self.ranges, (sel.stop-sel.start)/sel.step)
			if (sel.stop-sel.start)*sel.step < 0: return Rangelist(np.zeros([0,2],dtype=int),0)
			if sel.step > 0:
				return Rangelist(slice_helper(self.ranges, sel),(sel.stop-sel.start)/sel.step)
			else:
				res = slice_helper(self.n-self.ranges[::-1,::-1], slice(sel.stop+1, sel.start+1, -sel.step))
				return Rangelist(res, (sel.stop-sel.start)/sel.step)
		else:
			# Assume number
			i = np.searchsorted(self.ranges[:,0], sel, side="right")
			if i == 0: return False
			return self.ranges[i-1,0] <= sel and self.ranges[i-1,1] > sel
	@staticmethod
	def empty(nsamp):
		return Rangelist(np.zeros([0,2],dtype=int),n=nsamp,copy=False)
	@staticmethod
	def ones(nsamp):
		return Rangelist(np.array([[0,nsamp]],dtype=int),n=nsamp,copy=False)
	def sum(self): return np.sum(self.ranges[:,1]-self.ranges[:,0])
	# In numpy 1.11+, arrays of Rangelists become hard to construct if
	# __len__ is defined, as numpy tries to iterate through the Rangelist
	# as a sequence.
	#def __len__(self): return self.n
	def __repr__(self): return "Rangelist("+str(self.ranges)+",n="+repr(self.n)+")"
	def __str__(self): return repr(self)
	def copy(self): return Rangelist(self.ranges, self.n, copy=True)
	def invert(self):
		pad = np.vstack([[[0,0]],self.ranges,[[self.n,self.n]]])
		res = np.array([pad[:-1,1],pad[1:,0]]).T
		res = np.delete(res, np.where(res[:,1]==res[:,0]),0)
		return Rangelist(res, self.n)
	def to_mask(self):
		res = np.zeros(self.n,dtype=bool)
		for r1,r2 in self.ranges: res[r1:r2] = True
		return res
	def clear(self): self.ranges = self.ranges[0:0]
	def __add__(self, rlist):
		if isinstance(rlist, Multirange):
			return rlist + self
		else:
			return Rangelist(np.concatenate([self.ranges, Rangelist(rlist,self.n).ranges],0), self.n)
	def widen(self, n):
		n = np.zeros(2,dtype=int)+n
		if np.all(n == 0): return self
		ranges = self.ranges.copy()
		ranges[:,0] = np.maximum(ranges[:,0]-n[0], 0)
		ranges[:,1] = np.minimum(ranges[:,1]+n[1], self.n)
		return Rangelist(ranges, self.n, copy=False)

class Multirange:
	"""Multirange makes it easier to work with large numbers of rangelists.
	It is essentially a numpy array (though it does not expose the same
	functions) of such lists, but defines coherent slicing for both its own
	and the contained Rangelist objects indices."""
	def __init__(self, rangelists, copy=True):
		# Todo: Handle (neach, flat) inputs
		if rangelists is None:
			self.data = np.zeros([],dtype=np.object)
		if isinstance(rangelists, Multirange):
			if copy: rangelists = rangelists.copy()
			self.data = rangelists.data
		elif isinstance(rangelists, tuple):
			n, neach, flat = rangelists
			ncum = cumsum(neach,True)
			self.data = np.asarray([Rangelist(flat[a:b],n) for a,b in zip(ncum[:-1],ncum[1:])])
		else:
			# List or array input. Constructing directly via array constructor
			# is suddenly broken - it tries to iterate through every index
			if copy: rangelists = np.array(rangelists)
			self.data = np.asarray(rangelists)
	def __getitem__(self, sel):
		sel1, sel2 = split_slice(sel, [self.data.ndim,1])
		res = self.data[sel1]
		if isinstance(res, Rangelist): return res[sel2]
		res = res.copy()
		rflat = res.reshape(res.size)
		for i in xrange(rflat.size):
			rflat[i] = rflat[i][sel2]
		if rflat.size > 0 and not isinstance(rflat[0], Rangelist):
			return res.astype(bool)
		return Multirange(res, copy=False)
	@staticmethod
	def empty(ndet, nsamp):
		return Multirange([Rangelist.empty(nsamp) for det in xrange(ndet)])
	@staticmethod
	def ones(ndet,nsamp):
		return Multirange([Rangelist.ones(nsamp) for det in xrange(ndet)])
	def sum(self, flat=True):
		getsum = np.vectorize(lambda x: x.sum(), 'i')
		res = getsum(self.data)
		return np.sum(res) if flat else res
	@property
	def shape(self):
		if self.data.size > 0:
			return self.data.shape + (self.data.reshape(-1)[0].n,)
		else:
			return self.data.shape + (0,)
	@property
	def size(self): return np.product(self.shape)
	def copy(self): return Multirange(self.data, copy=True)
	def invert(self):
		return Multirange(np.vectorize(lambda x: x.invert(),'O')(self.data))
	def __repr__(self): return "Multirange("+str(self.data)+")"
	def __str__(self): return repr(self)
	def flatten(self):
		getlens = np.vectorize(lambda x: len(x.ranges), 'i')
		neach   = getlens(self.data)
		flat    = np.concatenate([r.ranges for r in self.data.reshape(self.data.size)])
		n       = self.data[0].n
		return n, neach, flat
	def extract(self, arr):
		"""Extract the samples corresponding to this Multirange from the array
		arr, returning them as a 1d array."""
		res = np.zeros(self.size, arr.dtype)
		i = 0
		for d, a in zip(self.data.reshape(-1), arr.reshape(-1,arr.shape[-1])):
			for r in d.ranges:
				n = r[1]-r[0]
				res[i:i+n] = a[r[0]:r[1]]
				i += n
		return res
	def insert(self, arr, vals):
		"""Reverse of extract"""
		i = 0
		for d, a in zip(self.data.reshape(-1), arr.reshape(-1,arr.shape[-1])):
			for r in d.ranges:
				n = r[1]-r[0]
				a[r[0]:r[1]] = vals[i:i+n]
				i += n
	def to_mask(self):
		dflat = self.data.reshape(self.data.size)
		res   = np.zeros([dflat.size, dflat[0].n],dtype=bool)
		for i, d in enumerate(dflat):
			res[i] = d.to_mask()
		return res.reshape(self.data.shape+(-1,))
	@staticmethod
	def from_mask(mask):
		fmask = mask.reshape(-1, mask.shape[-1])
		data  = np.empty(fmask.shape[0],dtype=object)
		for i in range(len(fmask)):
			data[i] = Rangelist(fmask[i])
		data = data.reshape(mask.shape[:-1])
		return Multirange(data)
	def clear(self):
		for d in self.data: d.clear()
	def __add__(self, rlist):
		if isinstance(rlist, Multirange):
			return Multirange([a+b for a,b in zip(self.data, rlist.data)])
		else:
			return Multirange([a+rlist for a in self.data])
	def widen(self, n):
		if np.all(np.asarray(n) == 0): return self
		return Multirange([d.widen(n) for d in self.data], copy=False)

def zeros(shape):
	assert(len(shape)==2)
	ranges = [Rangelist(np.zeros((0,2),dtype=int),shape[1]) for i in range(shape[0])]
	return Multirange(ranges)

def slice_helper(ranges, sel):
	"""Helper function for rangelist slicing. Gets an expanded slice with positive
	step size."""
	if len(ranges) == 0: return ranges
	res = ranges.copy()
	# Find the first range partially ahead of this point
	i = np.searchsorted(ranges[:,1], sel.start, side="right")
	if i < len(ranges):
		res[i,0] = max(sel.start, res[i,0])
	# and similarly for the end
	j = np.searchsorted(ranges[:,0], sel.stop, side="left")
	if j > 0:
		res[j-1,1] = min(sel.stop, res[j-1,1])
	res = res[i:j]
	res -= sel.start
	# Prioritize in-range vs. out-range when reducing resolution.
	# This means that we round the lower bounds down and the upper
	# bounds up.
	res[:,0] /= sel.step
	res[:,1] = (res[:,1]+sel.step-1)/sel.step
	# However, avoid rounding beyond the new edge of the TOD
	n_new = (sel.stop-sel.start)/sel.step
	res[:,1] = np.minimum(res[:,1],n_new)
	# Normalize ranges, merging overlapping ones
	res = range_union(res)
	# Prune empty ranges
	res = res[res[:,1]-res[:,0]>0]
	return res

def multify(f):
	"""Takes any function that operates on a 1d array and a Rangelist
	and returns a function that will do the same operation on a n+1 D
	array and an N-dimensional Multirange. The inplace argument of hte
	resulting function determines whether to modify the array argument
	or not."""
	def multif(arr, multi, inplace=False, *args, **kwargs):
		kwargs["inplace"] = inplace
		if isinstance(multi, Multirange):
			mflat  = multi.data.reshape(multi.data.size)
			aflat  = arr.reshape(np.prod(arr.shape[:-1]),arr.shape[-1])
			if inplace:
				for i in range(len(aflat)):
					f(aflat[i], mflat[i], *args, **kwargs)
				return arr
			else:
				# Determine the shape of the output
				res0 = f(aflat[0].copy(), mflat[0], *args, **kwargs)
				oaflat = np.empty((aflat.shape[0],)+res0.shape)
				oaflat[0] = res0
				for i in range(1,len(aflat)):
					oaflat[i] = f(aflat[i], mflat[i], *args, **kwargs)
				return oaflat.reshape(arr.shape[:-1]+res0.shape)
		else:
			return f(arr, multi, *args, **kwargs)
	multif.__doc__ = "Multified version of function with docstring:\n" + f.__doc__
	return multif

def stack_ranges(multiranges, axis=0):
	"""Return a multirange which is the result of stacking the input
	multiranges along the selected (non-sample) axis."""
	return Multirange(np.concatenate([m.data for m in multiranges],axis))
