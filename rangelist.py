"""This module provides classes for representing ranges of true and false values,
providing both a mask-like (numpy bool array) and list of from:to interface.
It also provides a convenience class for handling multiple of these range lists."""
import numpy as np
from enlib.slice import expand_slice, split_slice

class Rangelist:
	def __init__(self, ranges, n, copy=True):
		if copy: ranges = np.array(ranges)
		self.n      = n
		self.ranges = np.asarray(ranges)
	def __getitem__(self, sel):
		"""This function operates on the rangelist as if it were a dense numpy array.
		It returns either a sliced Rangelist or a bool."""
		if isinstance(sel,tuple):
			if len(sel) > 1: raise IndexError("Too many indices to Rangelist (only 1 supported)")
			if len(sel) == 0: return self
			sel = sel[0]
		if isinstance(sel,slice):
			sel = expand_slice(sel, self.n)
			if len(self.ranges) == 0: return self
			if (sel.stop-sel.start)*sel.step < 0: return Rangelist(np.zeros([0,2]),0)
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
	def sum(self): return np.sum(self.ranges[:,1]-self.ranges[:,0])
	def __len__(self): return self.n
	def __repr__(self): return "Rangelist("+str(self.ranges)+",n="+repr(self.n)+")"
	def __str__(self): return repr(self)
	def copy(self): return Rangelist(self.ranges, self.n, copy=True)

class Multirange:
	"""Multirange makes it easier to work with large numbers of rangelists.
	It is essentially a numpy array (though it does not expose the same
	functions) of such lists, but defines coherent slicing for both its own
	and the contained Rangelist objects indices."""
	def __init__(self, rangelists, copy=True):
		# Todo: Handle (neach, flat) inputs
		if isinstance(rangelists, Multirange):
			if copy: rangelists = rangelists.copy()
			self.data = rangelists.data
		else:
			if copy: rangelists = np.array(rangelists)
			self.data = np.asarray(rangelists)
	def __getitem__(self, sel):
		sel1, sel2 = split_slice(sel, [self.data.ndim,1])
		res = self.data[sel1]
		if isinstance(res, Rangelist): return res
		res = res.copy()
		rflat = res.reshape(res.size)
		for i in xrange(rflat.size):
			rflat[i] = rflat[i][sel2]
		if rflat.size > 0 and not isinstance(rflat[0], Rangelist):
			return res.astype(bool)
		return Multirange(res, copy=False)
	def sum(self, flat=True):
		getsum = np.vectorize(lambda x: x.sum(), 'i')
		res = getsum(self.data)
		return np.sum(res) if flat else res
	def copy(self): return Multirange(self.data, copy=True)
	def __repr__(self): return "Multirange("+str(self.data)+")"
	def __str__(self): return repr(self)
	def flatten(self):
		getlens = np.vectorize(lambda x: len(x.ranges), 'i')
		neach   = getlens(self.data)
		flat    = np.concatenate([r.ranges for r in self.data.reshape(self.data.size)])
		return neach, flat

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
	# Prune empty ranges
	res = res[res[:,1]-res[:,0]>0]
	return res
