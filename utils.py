import numpy as np

def lines(file_or_fname):
	"""Iterates over lines in a file, which can be specified
	either as a filename or as a file object."""
	if isinstance(file_or_fname, basestring):
		with open(file_or_fname,"r") as file:
			for line in file: yield line
	else:
		for line in file: yield line

def listsplit(seq, elem):
	inds = [i for i,v in enumerate(seq) if v == elem]
	ranges = zip([0]+[i+1 for i in inds],inds+[len(seq)])
	return [seq[a:b] for a,b in ranges]

def common_inds(arrs):
	"""Given a list of arrays, returns the indices into each of them of
	their common elements. For example
	  common_inds([[1,2,3,4,5],[2,4,6,8]]) -> [[1,3],[0,1]]"""
	inter = arrs[0]
	for arr in arrs[1:]:
		inter = np.lib.arraysetops.intersect1d(inter,arr)
	return [np.where(np.in1d(arr,inter))[0] for arr in arrs]

def split_slice(sel, ndims):
	"""Splits a numpy-compatible slice "sel" into sub-slices sub[:], such that
	a[sel] = s[sub[0]][:,sub[1]][:,:,sub[2]][...], This is useful when
	implementing arrays with heterogeneous indices."""
	if not isinstance(sel,tuple): sel = (sel,)
	# It's easy if we don't have ellipsis
	if Ellipsis not in sel: return split_slice_simple(sel, ndims)
	# Otherwise, fill in indices from the left and right...
	left, right = listsplit(sel, Ellipsis)
	resL = split_slice_simple(left,  ndims)
	resR = [v[::-1] for v in split_slice_simple(right[::-1], ndims[::-1])[::-1]]
	# and combine them.
	def combine(a, b, n):
		if len(b) == 0: return a
		if len(a) + len(b) < n:
			return a + (Ellipsis,) + b
		elif len(a) + len(b) == n:
			return a + b
		else:
			raise IndexError("Too many indices in ndim=%d array: "%n + str(a+b))
	return [combine(L,R,n) for L,R,n in zip(resL,resR,ndims)]

def split_slice_simple(sel, ndims):
	"""Helper function for split_slice. Splits a slice
	in the absence of ellipsis."""
	res = [[] for n in ndims]
	notNone = [v != None for v in sel]
	subs = np.concatenate([[0],cumsplit(notNone, ndims)])
	for i, r in enumerate(res):
		r += sel[subs[i]:subs[i+1]]
	return [tuple(v) for v in res]

def expand_slice(sel, n):
	"""Expands defaults and negatives in a slice to their implied values.
	After this, all entries of the slice are guaranteed to be present in their final form.
	Note, doing this twice may result in odd results, so don't send the result of this
	into functions that expect an unexpanded slice."""
	step = sel.step or 1
	def cycle(i,n): return min(i,n) if i >= 0 else n+i
	if step == 0: raise ValueError("slice step cannot be zero")
	if step > 0: return slice(cycle(sel.start or 0,n),cycle(sel.stop or n,n),step)
	else: return slice(cycle(sel.start or n-1, n), cycle(sel.stop,n) if sel.stop else -1, step)
