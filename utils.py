import numpy as np, scipy.ndimage

def lines(file_or_fname):
	"""Iterates over lines in a file, which can be specified
	either as a filename or as a file object."""
	if isinstance(file_or_fname, basestring):
		with open(file_or_fname,"r") as file:
			for line in file: yield line
	else:
		for line in file: yield line

def listsplit(seq, elem):
	"""Analogue of str.split for lists.
	listsplit([1,2,3,4,5,6,7],4) -> [[1,2],[3,4,5,6]]."""
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

def dict_apply_listfun(dict, function):
	"""Applies a function that transforms one list to another
	with the same number of elements to the values in a dictionary,
	returning a new dictionary with the same keys as the input
	dictionary, but the values given by the results of the function
	acting on the input dictionary's values. I.e.
	if f(x) = x[::-1], then dict_apply_listfun({"a":1,"b":2},f) = {"a":2,"b":1}."""
	keys = dict.keys()
	vals = [dict[key] for key in keys]
	res  = function(vals)
	return {key: res[i] for i, key in enumerate(keys)}

def unwind(a, period=2*np.pi):
	"""Given a list of angles or other cyclic coordinates
	where a and a+period have the same physical meaning,
	make a continuous by removing any sudden jumps due to
	period-wrapping. I.e. [0.07,0.02,6.25,6.20] would
	become [0.07,0.02,-0.03,-0.08] with the default period
	of 2*pi."""
	res = np.array(a)
	res[...,0] = rewind(res[...,0],ref=0,period=period)
	res[...,1:] -= np.cumsum(np.round((res[...,1:]-res[...,:-1])/period),-1)*period
	return res

def rewind(a, ref=0, period=2*np.pi):
	"""Given a list of angles or other cyclic corodinates,
	add or subtract multiples of the period in order to ensure
	that they all lie within the same period. The ref argument
	specifies the angle furthest away from the cut, i.e. the
	period cut will be at ref+period/2."""
	return ref + (a-ref+period/2)%period - period/2

def cumsplit(sizes, capacities):
	"""Given a set of sizes (of files for example) and a set of capacities
	(of disks for example), returns the index of the sizes for which
	each new capacity becomes necessary, assuming sizes can be split
	across boundaries.
	For example cumsplit([1,1,2,0,1,3,1],[3,2,5]) -> [2,5]"""
	return np.searchsorted(np.cumsum(sizes),np.cumsum(capacities),side="right")

def mask2range(mask):
	"""Convert a binary mask [True,True,False,True,...] into
	a set of ranges [:,{start,stop}]."""
	# We consider the outside of the array to be False
	mask  = np.concatenate([[False],mask,[False]]).astype(np.int8)
	# Find where we enter and exit ranges with true mask
	dmask = mask[1:]-mask[:-1]
	start = np.where(dmask>0)[0]
	stop  = np.where(dmask<0)[0]
	return np.array([start,stop]).T

def repeat_filler(d, n):
	"""Form an array n elements long by repeatedly concatenating
	d and d[::-1]."""
	d = np.concatenate([d,d[::-1]])
	nmul = (n+d.size-1)/d.size
	dtot = np.concatenate([d]*nmul)
	return dtot[:n]

def deslope(d, w=1, inplace=False):
	"""Remove a slope and mean from d, matching up the beginning
	and end of d. The w parameter controls the number of samples
	from each end of d that is used to determine the value to
	match up."""
	if not inplace: d = np.array(d)
	dflat = d.reshape(np.prod(d.shape[:-1]),d.shape[-1])
	for di in dflat:
		di -= np.arange(di.size)*(np.mean(di[-w:])-np.mean(di[:w]))/di.size+np.mean(di[:w])
	return d

def ctime2mjd(ctime):
	"""Converts from unix time to modified julian date."""
	return ctime/86400 + 40587.0

def mjd2ctime(mjd):
	"""Converts from modified julian date to unix time"""
	return (mjd-40587.0)*86400

def medmean(x, frac=0.5):
	x = np.sort(x)
	i = int(x.size*frac)/2
	return np.mean(x[i:-i])

def moveaxis(a, o, n):
	if o < 0: o = o+a.ndim
	if n < 0: n = n+a.ndim
	if n <= o: return np.rollaxis(a, o, n)
	else: return np.rollaxis(a, o, n+1)

def moveaxes(a, old, new):
	"""Move the axes listed in old to the positions given
	by new. This is like repeated calls to numpy rollaxis
	while taking into account the effect of previous rolls.

	This version is slow but simple and safe. It moves
	all axes to be moved to the end, and then moves them
	one by one to the target location."""
	# The final moves will happen in left-to-right order.
	# Hence, the first moves must be in the reverse of
	# this order.
	n = len(old)
	old   = np.asarray(old)
	order = np.argsort(new)
	rold  = old[order[::-1]]
	for i in range(n):
		a = moveaxis(a, rold[i], -1)
		# This may have moved some of the olds we're going to
		# move next, so update these
		for j in range(i+1,n):
			if rold[j] > rold[i]: rold[j] -= 1
	# Then do the final moves
	for i in range(n):
		a = moveaxis(a, -1, new[order[i]])
	return a
def partial_flatten(a, axes=[-1], pos=0):
	"""Flatten all dimensions of a except those mentioned
	in axes, and put the flattened one at the given position.
	The result is always at least 2d.

	Example: if a.shape is [1,2,3,4],
	then partial_flatten(a,[-1],0).shape is [6,4]."""
	a = moveaxes(a, axes, range(len(axes)))
	a = np.reshape(a, list(a.shape[:len(axes)])+[np.prod(a.shape[len(axes):])])
	return moveaxis(a, -1, pos)

def partial_expand(a, shape, axes=[-1], pos=0):
	"""Undo a partial flatten. Shape is the shape of the
	original array before flattening, and axes and pos should be
	the same as those passed to the flatten operation."""
	a = moveaxis(a, pos, -1)
	axes = np.array(axes)%len(shape)
	rest = list(np.delete(shape, axes))
	a = np.reshape(a, list(a.shape[:len(axes)])+rest)
	return moveaxes(a, range(len(axes)), axes)

def interpol(a, inds, order=3, mode="nearest", mask_nan=True):
	"""Given an array a[{x},{y}] and a list of
	float indices into a, inds[len(y),{z}],
	returns interpolated values at these positions
	as [{x},{z}]."""
	npre = a.ndim - inds.shape[0]
	res = np.empty(a.shape[:npre]+inds.shape[1:])
	fa, fr = partial_flatten(a, range(npre,a.ndim)), partial_flatten(res, range(npre, res.ndim))
	if mask_nan:
		mask = ~np.isfinite(fa)
		fa[mask] = 0
	for i in range(fa.shape[0]):
		fr[i] = scipy.ndimage.map_coordinates(fa[i], inds, order=order, mode=mode)
	if mask_nan and np.sum(mask) > 0:
		for i in range(mask.shape[0]):
			mask[i] = scipy.ndimage.map_coordinates(mask[i], inds, order=0, mode=mode)
		fa[mask] = np.nan
	return res
