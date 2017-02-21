import numpy as np, scipy.ndimage, os, errno, scipy.optimize, time, datetime, warnings, re

degree = np.pi/180
arcmin = degree/60
arcsec = arcmin/60
fwhm   = 1.0/(8*np.log(2))**0.5
T_cmb = 2.73
c = 299792458.0
h = 6.62606957e-34
k = 1.3806488e-23

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
	# Sadly, numpy arrays misbehave, and must be treated specially
	def iseq(e1, e2): return np.all(e1==e2)
	inds = [i for i,v in enumerate(seq) if iseq(v,elem)]
	ranges = zip([0]+[i+1 for i in inds],inds+[len(seq)])
	return [seq[a:b] for a,b in ranges]

def find(array, vals):
	"""Return the indices of each value of vals in the given array."""
	order = np.argsort(array)
	return order[np.searchsorted(array, vals, sorter=order)]

def contains(array, vals):
	"""Given an array[n], returns a boolean res[n], which is True
	for any element in array that is also in vals, and False otherwise."""
	array = np.asarray(array)
	vals  = np.sort(vals)
	inds  = np.searchsorted(vals, array)
	# If a value would be inserted after the end, it wasn't
	# present in the original array.
	inds[inds>=len(vals)] = 0
	return vals[inds] == array

def common_vals(arrs):
	"""Given a list of arrays, returns their intersection.
	For example
	  common_vals([[1,2,3,4,5],[2,4,6,8]]) -> [2,4]"""
	res = arrs[0]
	for arr in arrs[1:]:
		res = np.intersect1d(res,arr)
	return res

def common_inds(arrs):
	"""Given a list of arrays, returns the indices into each of them of
	their common elements. For example
	  common_inds([[1,2,3,4,5],[2,4,6,8]]) -> [[1,3],[0,1]]"""
	vals = common_vals(arrs)
	return [find(arr, vals) for arr in arrs]

def union(arrs):
	"""Given a list of arrays, returns their union."""
	res = arrs[0]
	for arr in arrs[1:]:
		res = np.union1d(res,arr)
	return res

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

def unwind(a, period=2*np.pi, axes=[-1], ref=0):
	"""Given a list of angles or other cyclic coordinates
	where a and a+period have the same physical meaning,
	make a continuous by removing any sudden jumps due to
	period-wrapping. I.e. [0.07,0.02,6.25,6.20] would
	become [0.07,0.02,-0.03,-0.08] with the default period
	of 2*pi."""
	res = rewind(a, period=period, ref=ref)
	for axis in axes:
		with flatview(res, axes=[axis]) as flat:
			# Avoid trying to sum nans
			mask = ~np.isfinite(flat)
			bad = flat[mask]
			flat[mask] = 0
			flat[:,1:]-= np.cumsum(np.round((flat[:,1:]-flat[:,:-1])/period),-1)*period
			# Restore any nans
			flat[mask] = bad
	return res

def rewind(a, ref=0, period=2*np.pi):
	"""Given a list of angles or other cyclic corodinates,
	add or subtract multiples of the period in order to ensure
	that they all lie within the same period. The ref argument
	specifies the angle furthest away from the cut, i.e. the
	period cut will be at ref+period/2."""
	a = np.asanyarray(a)
	if ref is "auto": ref = np.sort(a.reshape(-1))[a.size/2]
	return ref + (a-ref+period/2.)%period - period/2.

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

def deslope(d, w=1, inplace=False, axis=-1):
	"""Remove a slope and mean from d, matching up the beginning
	and end of d. The w parameter controls the number of samples
	from each end of d that is used to determine the value to
	match up."""
	if not inplace: d = np.array(d)
	with flatview(d, axes=[axis]) as dflat:
		for di in dflat:
			di -= np.arange(di.size)*(np.mean(di[-w:])-np.mean(di[:w]))/(di.size-1)+np.mean(di[:w])
	return d

def ctime2mjd(ctime):
	"""Converts from unix time to modified julian date."""
	return np.asarray(ctime)/86400. + 40587.0
def mjd2ctime(mjd):
	"""Converts from modified julian date to unix time."""
	return (np.asarray(mjd)-40587.0)*86400
day2sec = 86400.

def mjd2ctime(mjd):
	"""Converts from modified julian date to unix time"""
	return (np.asarray(mjd)-40587.0)*86400

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

	Example: if a.shape is [1,2,3,4],
	then partial_flatten(a,[-1],0).shape is [6,4]."""
	# Move the selected axes first
	a = moveaxes(a, axes, range(len(axes)))
	# Flatten all the other axes
	a = a.reshape(a.shape[:len(axes)]+(-1,))
	# Move flattened axis to the target position
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

def addaxes(a, axes):
	axes = np.array(axes)
	axes[axes<0] += a.ndim
	axes = np.sort(axes)[::-1]
	inds = [slice(None) for i in a.shape]
	for ax in axes: inds.insert(ax, None)
	return a[inds]

def delaxes(a, axes):
	axes = np.array(axes)
	axes[axes<0] += a.ndim
	axes = np.sort(axes)[::-1]
	inds = [slice(None) for i in a.shape]
	for ax in axes: inds[ax] = 0
	return a[inds]

class flatview:
	"""Produce a read/writable flattened view of the given array,
	via with flatview(arr) as farr:
		do stuff with farr
	Changes to farr are propagated into the original array.
	Flattens all dimensions of a except those mentioned
	in axes, and put the flattened one at the given position."""
	def __init__(self, array, axes=[], mode="rwc", pos=0):
		self.array = array
		self.axes  = axes
		self.flat  = None
		self.mode  = mode
		self.pos   = pos
	def __enter__(self):
		self.flat = partial_flatten(self.array, self.axes, pos=self.pos)
		if "c" in self.mode:
			self.flat = np.ascontiguousarray(self.flat)
		return self.flat
	def __exit__(self, type, value, traceback):
		# Copy back out from flat into the original array, if necessary
		if "w" not in self.mode: return
		if np.may_share_memory(self.array, self.flat): return
		# We need to copy back out
		self.array[:] = partial_expand(self.flat, self.array.shape, self.axes, pos=self.pos)

class nowarn:
	"""Use in with block to suppress warnings inside that block."""
	def __enter__(self):
		self.filters = list(warnings.filters)
		warnings.filterwarnings("ignore")
		return self
	def __exit__(self, type, value, traceback):
		warnings.filters = self.filters

def dedup(a):
	"""Removes consecutive equal values from a 1d array, returning the result.
	The original is not modified."""
	return a[np.concatenate([[True],a[1:]!=a[:-1]])]

def interpol(a, inds, order=3, mode="nearest", mask_nan=True, cval=0.0, prefilter=True):
	"""Given an array a[{x},{y}] and a list of float indices into a,
	inds[len(y),{z}], returns interpolated values at these positions as [{x},{z}]."""
	a    = np.asanyarray(a)
	inds = np.asanyarray(inds)
	inds_orig_nd = inds.ndim
	if inds.ndim == 1: inds = inds[:,None]

	npre = a.ndim - inds.shape[0]
	res = np.empty(a.shape[:npre]+inds.shape[1:],dtype=a.dtype)
	fa, fr = partial_flatten(a, range(npre,a.ndim)), partial_flatten(res, range(npre, res.ndim))
	if mask_nan:
		mask = ~np.isfinite(fa)
		fa[mask] = 0
	for i in range(fa.shape[0]):
		fr[i].real = scipy.ndimage.map_coordinates(fa[i].real, inds, order=order, mode=mode, cval=cval, prefilter=prefilter)
		if np.iscomplexobj(fa[i]):
			fr[i].imag = scipy.ndimage.map_coordinates(fa[i].imag, inds, order=order, mode=mode, cval=cval, prefilter=prefilter)
	if mask_nan and np.sum(mask) > 0:
		fmask = np.empty(fr.shape,dtype=bool)
		for i in range(mask.shape[0]):
			fmask[i] = scipy.ndimage.map_coordinates(mask[i], inds, order=0, mode=mode, cval=cval, prefilter=prefilter)
		fr[fmask] = np.nan
	if inds_orig_nd == 1: res = res[...,0]
	return res

def interpol_prefilter(a, npre=None, order=3, inplace=False):
	a = np.asanyarray(a)
	if not inplace: a = a.copy()
	if npre is None: npre = a.ndim - 2
	with flatview(a, range(npre, a.ndim), "rw") as aflat:
		for i in range(len(aflat)):
			aflat[i] = scipy.ndimage.spline_filter(aflat[i], order=order)
	return a

def bin_multi(pix, shape, weights=None):
	"""Simple multidimensional binning. Not very fast.
	Given pix[{coords},:] where coords are indices into an array
	with shape shape, count the number of hits in each pixel,
	returning map[shape]."""
	pix  = np.maximum(np.minimum(pix, (np.array(shape)-1)[:,None]),0)
	inds = np.ravel_multi_index(tuple(pix), tuple(shape))
	size = np.product(shape)
	if weights is not None: weights = inds*0+weights
	return np.bincount(inds, weights=weights, minlength=size).reshape(shape)

def grid(box, shape, endpoint=True, axis=0, flat=False):
	"""Given a bounding box[{from,to},ndim] and shape[ndim] in each
	direction, returns an array [ndim,shape[0],shape[1],...] array
	of evenly spaced numbers. If endpoint is True (default), then
	the end point is included. Otherwise, the last sample is one
	step away from the end of the box. For one dimension, this is
	similar to linspace:
		linspace(0,1,4)     =>  [0.0000, 0.3333, 0.6667, 1.0000]
		grid([[0],[1]],[4]) => [[0,0000, 0.3333, 0.6667, 1.0000]]
	"""
	n    = np.asarray(shape)
	box  = np.asfarray(box)
	off  = -1 if endpoint else 0
	inds = np.rollaxis(np.indices(n),0,len(n)+1) # (d1,d2,d3,...,indim)
	res  = inds * (box[1]-box[0])/(n+off) + box[0]
	if flat: res = res.reshape(-1, res.shape[-1])
	return np.rollaxis(res, -1, axis)

def cumsum(a, endpoint=False):
	"""As numpy.cumsum for a 1d array a, but starts from 0. If endpoint is True, the result
	will have one more element than the input, and the last element will be the sum of the
	array. Otherwise (the default), it will have the same length as the array, and the last
	element will be the sum of the first n-1 elements."""
	res = np.concatenate([[0],np.cumsum(a)])
	return res if endpoint else res[:-1]

def nearest_product(n, factors, direction="below"):
	"""Compute the highest product of positive integer powers of the specified
	factors that is lower than or equal to n. This is done using a simple,
	O(n) brute-force algorithm."""
	if 1 in factors: return n
	below = direction=="below"
	nmax = n+1 if below else n*min(factors)+1
	a = np.zeros(nmax+1,dtype=bool)
	a[1] = True
	best = 1
	for i in xrange(n+1):
		if not a[i]: continue
		for f in factors:
			m = i*f
			if below:
				if m > n: continue
			else:
				if m >= n: return m
			a[m] = True
			best = m
	return best

def mkdir(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

def decomp_basis(basis, vec):
	return np.linalg.solve(basis.dot(basis.T),basis.dot(vec.T)).T

def find_period(d, axis=-1):
	dwork = partial_flatten(d, [axis])
	guess = find_period_fourier(dwork)
	res = np.empty([3,len(dwork)])
	for i, (d1, g1) in enumerate(zip(dwork, guess)):
		res[:,i] = find_period_exact(d1, g1)
	periods = res[0].reshape(d.shape[:axis]+d.shape[axis:][1:])
	phases  = res[1].reshape(d.shape[:axis]+d.shape[axis:][1:])
	chisqs  = res[2].reshape(d.shape[:axis]+d.shape[axis:][1:])
	return periods, phases, chisqs

def find_period_fourier(d, axis=-1):
	"""This is a simple second-order estimate of the period of the
	assumed-periodic signal d. It finds the frequency with the highest
	power using an fft, and partially compensates for nonperiodicity
	by taking a weighted mean of the position of the top."""
	d2 = partial_flatten(d, [axis])
	fd  = np.fft.rfft(d2)
	ps = np.abs(fd)**2
	ps[:,0] = 0
	periods = []
	for p in ps:
		n = np.argmax(p)
		r = [int(n*0.5),int(n*1.5)+1]
		denom = np.sum(p[r[0]:r[1]])
		if denom <= 0: denom = 1
		n2 = np.sum(np.arange(r[0],r[1])*p[r[0]:r[1]])/denom
		periods.append(float(d.shape[axis])/n2)
	return np.array(periods).reshape(d.shape[:axis]+d.shape[axis:][1:])

def find_period_exact(d, guess):
	n = d.size
	# Restrict to at most 10 fiducial periods
	n = int(min(10,n/float(guess))*guess)
	off = (d.size-n)/2
	d = d[off:off+n]
	def chisq(x):
		w,phase = x
		model = interpol(d, np.arange(n)[None]%w+phase, order=1)
		return np.var(d-model)
	period,phase = scipy.optimize.fmin_powell(chisq, [guess,guess], xtol=1, disp=False)
	return period, phase+off, chisq([period,phase])/np.var(d**2)

def equal_split(weights, nbin):
	"""Split weights into nbin bins such that the total
	weight in each bin is as close to equal as possible.
	Returns a list of indices for each bin."""
	inds = np.argsort(weights)[::-1]
	bins = [[] for b in xrange(nbin)]
	bw   = np.zeros([nbin])
	for i in inds:
		j = np.argmin(bw)
		bins[j].append(i)
		bw[j] += weights[i]
	return bins

def range_sub(a,b, mapping=False):
	"""Given a set of ranges a[:,{from,to}] and b[:,{from,to}],
	return a new set of ranges c[:,{from,to}] which corresponds to
	the ranges in a with those in b removed. This might split individual
	ranges into multiple ones. If mapping=True, two extra objects are
	returned. The first is a mapping from each output range to the
	position in a it comes from. The second is a corresponding mapping
	from the set of cut a and b range to indices into a and b, with
	b indices being encoded as -i-1. a and b are assumed
	to be internally non-overlapping.

	Example: utils.range_sub([[0,100],[200,1000]], [[1,2],[3,4],[8,999]], mapping=True)
	(array([[   0,    1],
	        [   2,    3],
	        [   4,    8],
	        [ 999, 1000]]),
	array([0, 0, 0, 1]),
	array([ 0, -1,  1, -2,  2, -3,  3]))

	The last array can be interpreted as: Moving along the number line,
	we first encounter [0,1], which is a part of range 0 in c. We then
	encounter range 0 in b ([1,2]), before we hit [2,3] which is
	part of range 1 in c. Then comes range 1 in b ([3,4]) followed by
	[4,8] which is part of range 2 in c, followed by range 2 in b
	([8,999]) and finally [999,1000] which is part of range 3 in c.

	The same call without mapping: utils.range_sub([[0,100],[200,1000]], [[1,2],[3,4],[8,999]])
	array([[   0,    1],
	       [   2,    3],
	       [   4,    8],
	       [ 999, 1000]])
	"""
	def fixshape(a):
		a = np.asarray(a)
		if a.size == 0: a = np.zeros([0,2],dtype=int)
		return a
	a     = fixshape(a)
	b     = fixshape(b)
	ainds = np.argsort(a[:,0])
	binds = np.argsort(b[:,0])
	rainds= np.arange(len(a))[ainds]
	rbinds= np.arange(len(b))[binds]
	a = a[ainds]
	b = b[binds]
	ai,bi = 0,0
	c = []
	abmap = []
	rmap  = []
	while ai < len(a):
		# Iterate b until it ends past the start of a
		while bi < len(b) and b[bi,1] <= a[ai,0]:
			abmap.append(-rbinds[bi]-1)
			bi += 1
		# Now handle each b until the start of b is past the end of a
		pstart = a[ai,0]
		while bi < len(b) and b[bi,0] <= a[ai,1]:
			r=(pstart,min(a[ai,1],b[bi,0]))
			if r[1]-r[0] > 0:
				abmap.append(len(c))
				rmap.append(rainds[ai])
				c.append(r)
			abmap.append(-rbinds[bi]-1)
			pstart = b[bi,1]
			bi += 1
		# Then append what remains
		r=(pstart,a[ai,1])
		if r[1]>r[0]:
			abmap.append(len(c))
			rmap.append(rainds[ai])
			c.append(r)
		else:
			# If b extended beyond the end of a, then
			# we need to consider it again for the next a,
			# so undo the previous increment. This may lead to
			# the same b being added twice. We will handle that
			# by removing duplicates at the end.
			bi -= 1
		# And advance to the next range in a
		ai += 1
	c = np.array(c)
	# Remove duplicates if necessary
	abmap=dedup(np.array(abmap))
	rmap = np.array(rmap)
	return (c, rmap, abmap) if mapping else c

def range_union(a, mapping=False):
	"""Given a set of ranges a[:,{from,to}], return a new set where all
	overlapping ranges have been merged, where to >= from. If mapping=True,
	then the mapping from old to new ranges is also returned."""
	# We will make a single pass through a in sorted order
	a    = np.asarray(a)
	n    = len(a)
	inds = np.argsort(a[:,0])
	rmap = np.zeros(n,dtype=int)-1
	b    = []
	# i will point at the first unprocessed range
	for i in xrange(n):
		if rmap[inds[i]] >= 0: continue
		rmap[inds[i]] = len(b)
		start, end = a[inds[i]]
		# loop through every unprocessed range in range
		for j in xrange(i+1,n):
			if rmap[inds[j]] >= 0: continue
			if a[inds[j],0] > end: break
			# This range overlaps, so register it and merge
			rmap[inds[j]] = len(b)
			end = max(end, a[inds[j],1])
		b.append([start,end])
	b = np.array(b)
	if b.size == 0: b = b.reshape(0,2)
	return (b,rmap) if mapping else b

def range_normalize(a):
	"""Given a set of ranges a[:,{from,to}], normalize the ranges
	such that no ranges are empty, and all ranges go in increasing
	order. Decreasing ranges are interpreted the same way as in a slice,
	e.g. empty."""
	a = np.asarray(a)
	n1 = len(a)
	a = a[a[:,1]!=a[:,0]]
	reverse = a[:,1]<a[:,0]
	a = a[~reverse]
	n2 = len(a)
	return a

def range_cut(a, c):
	"""Cut range list a at positions given by c. For example
	range_cut([[0,10],[20,100]],[0,2,7,30,200]) -> [[0,2],[2,7],[7,10],[20,30],[30,100]]."""
	return range_sub(a,np.dstack([c,c])[0])

def compress_beam(sigma, phi):
	sigma = np.asarray(sigma,dtype=float)
	c,s=np.cos(phi),np.sin(phi)
	R = np.array([[c,-s],[s,c]])
	C = np.diag(sigma**-2)
	C = R.dot(C).dot(R.T)
	return np.array([C[0,0],C[1,1],C[0,1]])

def expand_beam(irads, return_V=False):
	C = np.array([[irads[0],irads[2]],[irads[2],irads[1]]])
	E, V = np.linalg.eigh(C)
	phi = np.arctan2(V[1,0],V[0,0])
	sigma = E**-0.5
	if sigma[1] > sigma[0]:
		sigma = sigma[::-1]
		phi += np.pi/2
	phi %= np.pi
	if return_V: return sigma, phi, V
	else: return sigma, phi

def combine_beams(irads_array):
	Cs = np.array([[[ir[0],ir[2]],[ir[2],ir[1]]] for ir in irads_array])
	Ctot = np.eye(2)
	for C in Cs:
		E, V = np.linalg.eigh(C)
		B = (V*E[None]**0.5).dot(V.T)
		Ctot = B.dot(Ctot).dot(B.T)
	return np.array([Ctot[0,0],Ctot[1,1],Ctot[0,1]])

def read_lines(fname, col=0):
	"""Read lines from file fname, returning them as a list of strings.
	If fname ends with :slice, then the specified slice will be applied
	to the list before returning."""
	toks = fname.split(":")
	fname, fslice = toks[0], ":".join(toks[1:])
	lines = [line.split()[col] for line in open(fname,"r") if line[0] != "#"]
	n = len(lines)
	return eval("lines"+fslice)

def loadtxt(fname):
	"""As numpy.loadtxt, but allows slice syntax."""
	toks = fname.split(":")
	fname, fslice = toks[0], ":".join(toks[1:])
	a = np.loadtxt(fname)
	return eval("a"+fslice)

def atleast_3d(a):
	a = np.asanyarray(a)
	if a.ndim == 0: return a.reshape(1,1,1)
	elif a.ndim == 1: return a.reshape(1,1,-1)
	elif a.ndim == 2: return a.reshape((1,)+a.shape)
	else: return a

def to_Nd(a, n, return_inverse=False):
	a = np.asanyarray(a)
	if n >= a.ndim:
		res = a.reshape((1,)*(n-a.ndim)+a.shape)
	else:
		res = a.reshape((-1,)+a.shape[1:])
	return (res, a.shape) if return_inverse else res

def between_angles(a, range, period=2*np.pi):
	a = rewind(a, np.mean(range), period=period)
	return (a>=range[0])&(a<range[1])

def greedy_split(data, n=2, costfun=max, workfun=lambda w,x: x if w is None else x+w):
	"""Given a list of elements data, return indices that would
	split them it into n subsets such that cost is approximately
	minimized. costfun specifies which cost to minimize, with
	the default being the value of the data themselves. workfun
	specifies how to combine multiple values. workfun(datum,workval)
	=> workval. scorefun then operates on a list of the total workval
	for each group score = scorefun([workval,workval,....]).

	Example: greedy_split(range(10)) => [[9,6,5,2,1,0],[8,7,4,3]]
	         greedy_split([1,10,100]) => [[2],[1,0]]
	         greedy_split("012345",costfun=lambda x:sum([xi**2 for xi in x]),
	          workfun=lambda w,x:0 if x is None else int(x)+w)
	          => [[5,2,1,0],[4,3]]
	"""
	# Sort data based on standalone costs
	costs = []
	nowork = workfun(None,None)
	work = [nowork for i in xrange(n)]
	for d in data:
		work[0] = workfun(nowork,d)
		costs.append(costfun(work))
	order = np.argsort(costs)[::-1]
	# Build groups using greedy algorithm
	groups = [[] for i in xrange(n)]
	work   = [nowork for i in xrange(n)]
	cost   = costfun(work)
	for di in order:
		d = data[di]
		# Try adding to each group
		for i in xrange(n):
			iwork = workfun(work[i],d)
			icost = costfun(work[:i]+[iwork]+work[i+1:])
			if i == 0 or icost < best[2]: best = (i,iwork,icost)
		# Add it to the best group
		i, iwork, icost = best
		groups[i].append(di)
		work[i] = iwork
		cost = icost
	return groups, cost, work

def cov2corr(C):
	"""Scale rows and columns of C such that its diagonal becomes one.
	This produces a correlation matrix from a covariance matrix. Returns
	the scaled matrix and the square root of the original diagonal."""
	std  = np.diag(C)**0.5
	istd = 1/std
	return np.einsum("ij,i,j->ij",C,istd,istd), std
def corr2cov(corr,std):
	"""Given a matrix "corr" and an array "std", return a version
	of corr with each row and column scaled by the corresponding entry
	in std. This is the reverse of cov2corr."""
	return np.einsum("ij,i,j->ij",corr,std,std)

def eigsort(A, nmax=None, merged=False):
	"""Return the eigenvalue decomposition of the real, symmetric matrix A.
	The eigenvalues will be sorted from largest to smallest. If nmax is
	specified, only the nmax largest eigenvalues (and corresponding vectors)
	will be returned. If merged is specified, E and V will not be returned
	separately. Instead, Q=VE**0.5 will be returned, such that QQ' = VEV'."""
	E,V  = np.linalg.eigh(A)
	inds = np.argsort(E)[::-1][:nmax]
	if merged: return V[:,inds]*E[inds][None]**0.5
	else:      return E[inds],V[:,inds]

def nodiag(A):
	"""Returns matrix A with its diagonal set to zero."""
	A = np.array(A)
	np.fill_diagonal(A,0)
	return A

def date2ctime(dstr):
	import dateutil.parser
	d = dateutil.parser.parse(dstr, ignoretz=True, tzinfos=0)
	return time.mktime(d.timetuple())

def bounding_box(boxes):
	"""Compute bounding box for a set of boxes [:,2,:], or a
	set of points [:,2]"""
	boxes = np.asarray(boxes)
	if boxes.ndim == 2:
		return np.array([np.min(boxes,0),np.max(boxes,0)])
	else:
		return np.array([np.min(boxes[:,0,:],0),np.max(boxes[:,1,:],0)])

def unpackbits(a): return np.unpackbits(np.atleast_1d(a).view(np.uint8)[::-1])[::-1]

def box2corners(box):
	"""Given a [{from,to},:] bounding box, returns [ncorner,:] coordinates
	of of all its corners."""
	box = np.asarray(box)
	ndim= box.shape[1]
	return np.array([[box[b,bi] for bi,b in enumerate(unpackbits(i)[:ndim])] for i in range(2**ndim)])

def box2contour(box, nperedge=5):
	"""Given a [{from,to},:] bounding box, returns [npoint,:] coordinates
	definiting its edges. Nperedge is the number of samples per edge of
	the box to use. For nperedge=2 this is equal to box2corners. Nperegege
	can be a list, in which case the number indicates the number to use in
	each dimension."""
	box      = np.asarray(box)
	ndim     = box.shape[1]
	nperedge = np.zeros(ndim,int)+nperedge
	# Generate the range of each coordinate
	points = []
	for i in range(ndim):
		x = np.linspace(box[0,i],box[1,i],nperedge[i])
		for j in range(2**ndim):
			bits = unpackbits(j)[:ndim]
			if bits[i]: continue
			y = np.zeros((len(x),ndim))
			y[:] = box[bits,np.arange(ndim)]; y[:,i] = x
			points.append(y)
	return np.concatenate(points,0)

def box_slice(a, b):
	"""Given two boxes/boxarrays of shape [{from,to},dims] or [:,{from,to},dims],
	compute the bounds of the part of each b that overlaps with each a, relative
	to the corner of a. For example box_slice([[2,5],[10,10]],[[0,0],[5,7]]) ->
	[[0,0],[3,2]]."""
	a  = np.asarray(a)
	b  = np.asarray(b)
	fa = a.reshape(-1,2,a.shape[-1])
	fb = b.reshape(-1,2,b.shape[-1])
	s  = np.minimum(np.maximum(0,fb[None,:]-fa[:,None,0,None]),fa[:,None,1,None]-fa[:,None,0,None])
	return s.reshape(a.shape[:-2]+b.shape[:-2]+(2,2))

def box_area(a):
	"""Compute the area of a [{from,to},ndim] box, or an array of such boxes."""
	return np.abs(np.product(a[...,1,:]-a[...,0,:],-1))

def box_overlap(a, b):
	"""Given two boxes/boxarrays, compute the overlap of each box with each other
	box, returning the area of the overlaps. If a is [2,ndim] and b is [2,ndim], the
	result will be a single number. if a is [n,2,ndim] and b is [2,ndim], the result
	will be a shape [n] array. If a is [n,2,ndim] and b is [m,2,ndim], the result will'
	be [n,m] areas."""
	return box_area(box_slice(a,b))

def widen_box(box, margin=1e-3, relative=True):
	box = np.asarray(box)
	margin = np.zeros(box.shape[1:])+margin
	if relative: margin = (box[1]-box[0])*margin
	margin = np.asarray(margin) # Support 1d case
	margin[box[0]>box[1]] *= -1
	return np.array([box[0]-margin/2, box[1]+margin/2])

def unwrap_range(range, nwrap=2*np.pi):
	"""Given a logically ordered range[{from,to},...] that
	may have been exposed to wrapping with period nwrap,
	undo the wrapping so that range[1] > range[0]
	but range[1]-range[0] is as small as possible.
	Also makes the range straddle 0 if possible.

	Unlike unwind and rewind, this function will not
	turn a very wide range into a small one because it
	doesn't assume that ranges are shorter than half the
	sky. But it still shortens ranges that are longer than
	a whole wrapping period."""
	range = np.asanyarray(range)
	range[1] -= np.floor((range[1]-range[0])/nwrap)*nwrap
	range    -= np.floor(range[1,None]/nwrap)*nwrap
	return range

def sum_by_id(a, ids, axis=0):
	ra = moveaxis(a, axis, 0)
	fa = ra.reshape(ra.shape[0],-1)
	fb = np.zeros((np.max(ids)+1,fa.shape[1]),fa.dtype)
	for i,id in enumerate(ids):
		fb[id] += fa[i]
	rb = fb.reshape((fb.shape[0],)+ra.shape[1:])
	return moveaxis(rb, 0, axis)

def allreduce(a, comm, op=None):
	res = a.copy()
	if op is None: comm.Allreduce(a, res)
	else:          comm.Allreduce(a, res, op)
	return res

def allgather(a, comm):
	a   = np.asarray(a)
	res = np.zeros((comm.size,)+a.shape,dtype=a.dtype)
	if np.issubdtype(a.dtype, str):
		comm.Allgather(a.view(dtype=np.uint8), res.view(dtype=np.uint8))
	else:
		comm.Allgather(a, res)
	return res

def allgatherv(a, comm, axis=0):
	"""Perform an mpi allgatherv along the specified axis of the array
	a, returning an array with the individual process arrays concatenated
	along that dimension. For example gatherv([[1,2]],comm) on one task
	and gatherv([[3,4],[5,6]],comm) on another task results in
	[[1,2],[3,4],[5,6]] for both tasks."""
	a  = np.asarray(a)
	fa = moveaxis(a, axis, 0)
	# mpi4py doesn't handle all types. But why not just do this
	# for everything?
	must_fix = np.issubdtype(a.dtype, str) or a.dtype == bool
	if must_fix:
		fa = fa.view(dtype=np.uint8)
	ra = fa.reshape(fa.shape[0],-1) if fa.size > 0 else fa.reshape(0,np.product(fa.shape[1:],dtype=int))
	N  = ra.shape[1]
	n  = allgather([len(ra)],comm)
	o  = cumsum(n)
	rb = np.zeros((np.sum(n),N),dtype=ra.dtype)
	comm.Allgatherv(ra, (rb, (n*N,o*N)))
	fb = rb.reshape((rb.shape[0],)+fa.shape[1:])
	# Restore original data type
	if must_fix:
		fb = fb.view(dtype=a.dtype)
	return moveaxis(fb, 0, axis)

def uncat(a, lens):
	"""Undo a concatenation operation. If a = np.concatenate(b)
	and lens = [len(x) for x in b], then uncat(a,lens) returns
	b."""
	cum = cumsum(lens, endpoint=True)
	return [a[cum[i]:cum[i+1]] for i in xrange(len(lens))]

def ang2rect(angs, zenith=True, axis=0):
	"""Convert a set of angles [{phi,theta},...] to cartesian
	coordinates [{x,y,z},...]. If zenith is True (the default),
	the theta angle will be taken to go from 0 to pi, and measure
	the angle from the z axis. If zenith is False, then theta
	goes from -pi/2 to pi/2, and measures the angle up from the xy plane."""
	phi, theta = moveaxis(angs, axis, 0)
	ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
	if zenith: res = np.array([st*cp,st*sp,ct])
	else:      res = np.array([ct*cp,ct*sp,st])
	return moveaxis(res, 0, axis)

def rect2ang(rect, zenith=True, axis=0):
	"""The inverse of ang2rect."""
	x,y,z = moveaxis(rect, axis, 0)
	r     = (x**2+y**2)**0.5
	phi   = np.arctan2(y,x)
	if zenith: theta = np.arctan2(r,z)
	else:      theta = np.arctan2(z,r)
	return moveaxis(np.array([phi,theta]), 0, axis)

def angdist(a, b, zenith=True, lim=1e-7, axis=0):
	"""Compute the angluar distance between a[{phi,theta},...] and
	b[{phi,theta},...], where the ... part can have different lengths
	as long as their transposes broadcast to each other."""
	a, b = np.asarray(a), np.asarray(b)
	ra = ang2rect(moveaxis(a, axis, 0), zenith, axis=0)
	rb = ang2rect(moveaxis(b, axis, 0), zenith, axis=0)
	c = np.sum(ra.T*rb.T,-1).T
	res = np.full(c.shape, np.NaN)
	res[c>=1-lim] = 0
	res[c<=-(1-lim)] = np.pi
	res[(c>-(1-lim))&(c<1-lim)] = np.arccos(c[(c>-(1-lim))&(c<1-lim)])
	return res

def rotmatrix(ang, raxis, axis=0):
	"""Construct a 3d rotation matrix representing a rotation of
	ang degrees around the specified rotation axis raxis, which can be "x", "y", "z"
	or 0, 1, 2. If ang is a scalar, the result will be [3,3]. Otherwise,
	it will be ang.shape + (3,3)."""
	ang  = np.asarray(ang)
	raxis = raxis.lower()
	c, s = np.cos(ang), np.sin(ang)
	R = np.zeros(ang.shape + (3,3))
	if   raxis == 0 or raxis == "x": R[...,0,0]=1;R[...,1,1]= c;R[...,1,2]=-s;R[...,2,1]= s;R[...,2,2]=c
	elif raxis == 1 or raxis == "y": R[...,0,0]=c;R[...,0,2]= s;R[...,1,1]= 1;R[...,2,0]=-s;R[...,2,2]=c
	elif raxis == 2 or raxis == "z": R[...,0,0]=c;R[...,0,1]=-s;R[...,1,0]= s;R[...,1,1]= c;R[...,2,2]=1
	else: raise ValueError("Rotation axis %s not recognized" % raxis)
	return moveaxis(R, 0, axis)

def label_unique(a, axes=(), rtol=1e-5, atol=1e-8):
	"""Given an array of values, return an array of
	labels such that all entries in the array with the
	same label will have approximately the same value.
	Labels count contiguously from 0 and up.
	axes specifies which axes make up the subarray that
	should be compared for equality. For scalars,
	use axes=()."""
	a = np.asarray(a)
	axes = [i % a.ndim for i in axes]
	rest = [s for i,s in enumerate(a.shape) if i not in axes]

	# First reshape into a doubly-flattened 2d array [nelem,ndim]
	fa = partial_flatten(a, axes, 0)
	fa = fa.reshape(np.product(rest),-1)
	# Can't use lexsort, as it has no tolerance. This
	# is O(N^2) instead of O(NlogN)
	id = 0
	ids = np.zeros(len(fa),dtype=int)-1
	for i,v in enumerate(fa):
		if ids[i] >= 0: continue
		match = np.all(np.isclose(v,fa,rtol=rtol,atol=atol),-1)
		ids[match] = id
		id += 1
	return ids.reshape(rest)

def transpose_inds(inds, nrow, ncol):
	"""Given a set of flattened indices into an array of shape (nrow,ncol),
	return the indices of the corresponding elemens in a transposed array."""
	row_major = inds
	row, col = row_major/ncol, row_major%ncol
	return col*nrow + row

def rescale(a, range=[0,1]):
	"""Rescale a such that min(a),max(a) -> range[0],range[1]"""
	mi, ma = np.min(a), np.max(a)
	return (a-mi)/(ma-mi)*(range[1]-range[0])+range[0]

def split_by_group(a, start, end):
	"""Split string a into non-group and group sections,
	where a group is defined as a set of characters from
	a start character to a corresponding end character."""
	res, ind, n = [], 0, 0
	new = True
	for c in a:
		if new:
			res.append("")
			new = False
		i = start.find(c)
		if n == 0:
			if i >= 0:
				# Start of new group
				res.append("")
				ind = i
				n += 1
		else:
			if start[ind] == c:
				n += 1
			elif end[ind] == c:
				n-= 1
				if n == 0: new = True
		res[-1] += c
	return res

def split_outside(a, sep, start="([{", end=")]}"):
	"""Split string a at occurences of separator sep, except when
	it occurs inside matching groups of start and end characters."""
	segments = split_by_group(a, start, end)
	res = [""]
	for seg in segments:
		if len(seg) == 0: continue
		if seg[0] in start:
			res[-1] += seg
		else:
			toks = seg.split(sep)
			res[-1] += toks[0]
			res += toks[1:]
	return res

def find_equal_groups(a, tol=0):
	"""Given a[nsamp,ndim], return groups[ngroup][{ind,ind,ind,...}]
	of indices into a for which all the values in the second index
	of a is the same. group_equal([[0,1],[1,2],[0,1]]) -> [[0,2],[1]]."""
	def calc_diff(a1,a2):
		if a1.dtype.char == 'S': return a1 != a2
		else: return a1-a2
	a = np.asarray(a)
	if a.ndim == 1: a = a[:,None]
	n = len(a)
	inds = np.argsort(a[:,0])
	done = np.full(n, False, dtype=bool)
	res = []
	for i in xrange(n):
		if done[i]: continue
		xi = inds[i]
		res.append([xi])
		done[i] = True
		for j in xrange(i+1,n):
			if done[j]: continue
			xj = inds[j]
			if calc_diff(a[xj,0], a[xi,0]) > tol:
				# Current group is done
				break
			if np.sum(calc_diff(a[xj],a[xi])**2) <= tol**2:
				# Close enough
				res[-1].append(xj)
				done[j] = True
	return res

def minmax(a, axis=None):
	"""Shortcut for np.array([np.min(a),np.max(a)]), since I do this
	a lot."""
	return np.array([np.min(a, axis=axis),np.max(a, axis=axis)])

def point_in_polygon(points, polys):
	"""Given a points[n|None,2] and a set of polys[n|None,nvertex,2], return
	inside[n|None].

	Examples:
	utils.point_in_polygon([0.5,0.5],[[0,0],[0,1],[1,1],[1,0]]) -> True
	utils.point_in_polygon([[0.5,0.5],[2,1]],[[0,0],[0,1],[1,1],[1,0]]) -> [True, False]
	"""
	# Make sure we have arrays, and that they have a floating point data type
	points = np.asarray(points)+0.0
	polys  = np.asarray(polys) +0.0
	npre   = max(points.ndim-1,polys.ndim-2)
	nvert  = polys.shape[-2]
	dirs   = np.zeros(max(polys.shape[0],points.shape[0]) if npre else (), dtype=np.int32)
	def direction(a,b): return np.sign(a[...,0]*b[...,1]-a[...,1]*b[...,0]).astype(np.int32)
	for i in range(nvert):
		v1 = polys[...,i-1,:]
		v2 = polys[...,i,:]
		dirs += direction(v2-v1, points-v1)
	inside = np.abs(dirs) == nvert
	return inside

def block_mean_filter(a, width):
	"""Perform a binwise smoothing of a, where all samples
	in each bin of the given width are replaced by the mean
	of the samples in that bin."""
	a = np.array(a)
	if a.shape[-1] < width:
		a[:] = np.mean(a,-1)[...,None]
	else:
		width  = int(width)
		nblock = (a.shape[-1]+width-1)/width
		apad   = np.concatenate([a,a[...,-2::-1]],-1)
		work   = apad[...,:width*nblock]
		work   = work.reshape(work.shape[:-1]+(nblock,width))
		work[:]= np.mean(work,-1)[...,None]
		work   = work.reshape(work.shape[:-2]+(-1,))
		a[:]   = work[...,:a.shape[-1]]
	return a

def ctime2date(timestamp, tzone=0, fmt="%Y-%m-%d"):
	return datetime.datetime.utcfromtimestamp(timestamp+tzone*3600).strftime(fmt)

def tofinite(arr, val=0):
	"""Return arr with all non-finite values replaced with val."""
	arr = np.asanyarray(arr).copy()
	if arr.ndim == 0:
		if ~np.isfinite(arr): arr = val
	else:
		arr[~np.isfinite(arr)] = val
	return arr

def parse_ints(s): return parse_numbers(s, int)
def parse_floats(s): return parse_numbers(s, float)
def parse_numbers(s, dtype=None):
	res = []
	for word in s.split(","):
		toks = [float(w) for w in word.split(":")]
		if ":" not in word:
			res.append(toks[:1])
		else:
			start, stop = toks[:2]
			step = toks[2] if len(toks) > 2 else 1
			res.append(np.arange(start,stop,step))
	res = np.concatenate(res)
	if dtype is not None:
		res = res.astype(dtype)
	return res

def triangle_wave(x, period=1):
	"""Return a triangle wave with amplitude 1 and the given period."""
	# This order (rather than x/period%1) gave smaller errors
	x = x % period / period * 4
	m1 = x < 1
	m2 = (x < 3) ^ m1
	m3 = x >= 3
	res = x.copy()
	res[m1] = x[m1]
	res[m2] = 2-x[m2]
	res[m3] = x[m3]-4
	return res

### Binning ####

def edges2bins(edges):
	res = np.zeros((edges.size-1,2),int)
	res[:,0] = edges[:-1]
	res[:,1] = edges[1:]
	return res

def bins2edges(bins):
	return np.concatenate([bins[:,0],bins[1,-1:]])

def linbin(n, nbin=None, nmin=None):
	"""Given a number of points to bin and the number of approximately
	equal-sized bins to generate, returns [nbin_out,{from,to}].
	nbin_out may be smaller than nbin. The nmin argument specifies
	the minimum number of points per bin, but it is not implemented yet.
	nbin defaults to the square root of n if not specified."""
	if not nbin: nbin = int(np.round(n**0.5))
	tmp  = np.arange(nbin+1)*n/nbin
	return np.vstack((tmp[:-1],tmp[1:])).T

def expbin(n, nbin=None, nmin=8, nmax=0):
	"""Given a number of points to bin and the number of exponentially spaced
	bins to generate, returns [nbin_out,{from,to}].
	nbin_out may be smaller than nbin. The nmin argument specifies
	the minimum number of points per bin. nbin defaults to n**0.5"""
	if not nbin: nbin = int(np.round(n**0.5))
	tmp  = np.array(np.exp(np.arange(nbin+1)*np.log(n+1)/nbin)-1,dtype=int)
	fixed = [tmp[0]]
	i = 0
	while i < nbin:
		for j in range(i+1,nbin+1):
			if tmp[j]-tmp[i] >= nmin:
				fixed.append(tmp[j])
				i = j
	# Optionally split too large bins
	if nmax:
		tmp = [fixed[0]]
		for v in fixed[1:]:
			dv = v-tmp[-1]
			nsplit = (dv+nmax-1)/nmax
			tmp += [tmp[-1]+dv*(i+1)/nsplit for i in range(nsplit)]
		fixed = tmp
	tmp = np.array(fixed)
	tmp[-1] = n
	return np.vstack((tmp[:-1],tmp[1:])).T

def bin_data(bins, d, op=np.mean):
	"""Bin the data d into the specified bins along the last dimension. The result has
	shape d.shape + (nbin,)."""
	nbin  = bins.shape[0]
	dflat = d.reshape(-1,d.shape[-1])
	dbin  = np.zeros([dflat.shape[0], nbin])
	for bi, b in enumerate(bins):
		dbin[:,bi] = op(dflat[:,b[0]:b[1]],1)
	return dbin.reshape(d.shape[:-1]+(nbin,))

def bin_expand(bins, bdata):
	res = np.zeros(bdata.shape[:-1]+(bins[-1,1],),bdata.dtype)
	for bi, b in enumerate(bins):
		res[...,b[0]:b[1]] = bdata[...,bi]
	return res

def is_int_valued(a): return a%1 == 0

#### Matrix operations that don't need fortran ####

# Don't do matmul - it's better expressed with einsum

def solve(A, b, axes=[-2,-1], masked=False):
	"""Solve the linear system Ax=b along the specified axes
	for A, and axes[0] for b. If masked is True, then entries
	where A00 along the given axes is zero will be skipped."""
	A,b = np.asanyarray(A), np.asanyarray(b)
	baxes = axes if A.ndim == b.ndim else [axes[0]%A.ndim]
	fA = partial_flatten(A, axes)
	fb = partial_flatten(b, baxes)
	if masked:
		mask = fA[...,0,0] != 0
		fb[~mask] = 0
		fb[mask]  = np.linalg.solve(fA[mask],fb[mask])
	else:
		fb = np.linalg.solve(fA,fb)
	return partial_expand(fb, b.shape, baxes)

def eigpow(A, e, axes=[-2,-1], rlim=None, alim=None):
	"""Compute the e'th power of the matrix A (or the last
	two axes of A for higher-dimensional A) by exponentiating
	the eigenvalues. A should be real and symmetric.

	When e is not a positive integer, negative eigenvalues
	could result in a complex result. To avoid this, negative
	eigenvalues are set to zero in this case.

	Also, when e is not positive, tiny eigenvalues dominated by
	numerical errors can be blown up enough to drown out the
	well-measured ones. To avoid this, eigenvalues
	smaller than 1e-13 for float64 or 1e-4 for float32 of the
	largest one (rlim), or with an absolute value less than 2e-304 for float64 or
	1e-34 for float32 (alim) are set to zero for negative e. Set alim
	and rlim to 0 to disable this behavior.
	"""
	# This function basically does
	# E,V = np.linalg.eigh(A)
	# E **= e
	# return (V*E).dot(V.T)
	# All the complicated stuff is there to support axes and tolerances.
	if axes[0]%A.ndim != A.ndim-2 or axes[1]%A.ndim != A.ndim-1:
		fa = partial_flatten(A, axes)
		fa = eigpow(fa, e, rlim=rlim, alim=alim)
		return partial_expand(fa, A.shape, axes)
	else:
		E, V = np.linalg.eigh(A)
		if rlim is None: rlim = np.finfo(E.dtype).resolution*100
		if alim is None: alim = np.finfo(E.dtype).tiny*1e4
		mask = np.full(E.shape, False, np.bool)
		if not is_int_valued(e):
			mask |= E < 0
		if e < 0:
			aE = np.abs(E)
			mask |= (aE < np.max(aE)*rlim) | (aE < alim)
		E[~mask] **= e
		E[mask]    = 0
		res = np.einsum("...ij,...kj->...ik",V*E[...,None,:],V)
		return res

def nint(a):
	"""Return a rounded to the nearest integer, as an integer."""
	return np.int0(np.round(a))

format_regex = r"%(\([a-zA-Z]\w*\)|\(\d+)\)?([ +0#-]*)(\d*|\*)(\.\d+|\.\*)?(ll|[lhqL])?(.)"
def format_to_glob(format):
	"""Given a printf format, construct a glob pattern that will match
	its outputs. However, since globs are not very powerful, the resulting
	glob will be much more premissive than the input format, and you will
	probably want to filter the results further."""
	# This matches a pretty general printf format
	def subfun(m):
		name, flags, width, prec, size, type = m.groups()
		if type == '%': return '%'
		else: return '*'
	return re.sub(format_regex, subfun, format)

def format_to_regex(format):
	"""Given a printf format, construct a regex that will match its outputs."""
	ireg = r"([^%]*)"+format_regex+r"([^%]*)"
	def subfun(m):
		pre, name, flags, width, prec, size, type, post = m.groups()
		opre  = re.escape(pre)
		opost = re.escape(post)
		open  = r"(?P<"+name[1:-1]+">" if name is not None else "("
		# Expand variable widths
		iwidth = 0 if width is None or width == '*' or width == '' else int(width)
		iprec  = 0 if prec  is None or prec  == '*' else int(prec[1:])
		if type == '%': return opre + '%' + opost
		if type == 's':
			if "-" in flags: return opre + open + ".*) *" + opost
			else:            return opre + r" *" + open + ".*)" + opost
		else:
			# Numeric type
			if   "+" in flags: omid = r"[+-]"
			elif " " in flags: omid = r"[ -]"
			else: omid = r"-?"
			if "-" in flags:
				prepad  = ""
				postpad = " *"
			else:
				prepad  = r"0*" if "0" in flags else r" *"
				postpad = ""
			if type in ['d','i','u'] or type in ['f','F'] and prec == '0':
				num = r"\d+"
			elif type == 'o': num = r"[0-7]+"
			elif type == 'x': num = r"[0-9a-f]+"
			elif type == 'X': num = r"[0-9A-F]+"
			elif type == 'f': num = r"\d+\.\d*"
			elif type == 'e': num = r"\d+\.\d*e[+-]\d+"
			elif type == 'E': num = r"\d+\.\d*E[+-]\d+"
			elif type == 'g': num = r"(\d+(\.\d*)?)|(\d+\.\d*e[+-]\d+)"
			elif type == 'G': num = r"(\d+(\.\d*)?)|(\d+\.\d*E[+-]\d+)"
			else: return NotImplementedError("Format character '%s'" % type)
			omid = prepad + open + omid + num + r")" + postpad
			return opre + omid + opost
	return re.sub(ireg, subfun, format)
