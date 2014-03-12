"""This module is intended to make it easier to implement slicing."""

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

def split_slice(sel, ndims):
	"""Splits a numpy-compatible slice "sel" into sub-slices sub[:], such that
	a[sel] = s[sub[0]][:,sub[1]][:,:,sub[2]][...], This is useful when
	implementing arrays with heterogeneous indices."""
	if not isinstance(sel,tuple): sel = (sel,)
	# It's easy if we don't have ellipsis.
	# What the heck? "in" operator is apparently broken for lists that
	# contain numpy arrays.
	if Ellipsis not in [type(i) for i in sel]: return split_slice_simple(sel, ndims)
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
