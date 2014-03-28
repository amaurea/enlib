import numpy as np, enlib.utils

def sym_compress(mat, which=None, n=None, scheme=None, axes=[0,1]):
	"""Extract the unique elements of a symmetric matrix, and
	return them as a flat array. For multidimensional arrays,
	the extra dimensions keep their shape. The optional argument
	'which' indicates the compression scheme, as returned by
	compressed_order. The optional argument 'n' indicates the
	number of elements to keep (the default is to keep all unique
	elements). The 'axes' argument indicates which axes to operate on."""
	if n is None: n = mat.shape[axes[0]]*(mat.shape[axes[0]]+1)/2
	if which==None: which = compressed_order(n, scheme)
	m = np.rollaxis(np.rollaxis(mat, axes[1]), axes[0])
	res = np.array([m[w[0],w[1]] for w in which])
	return np.rollaxis(res, 0, axes[0])

def sym_expand(mat, which=None, ncomp=None, scheme=None, axis=0):
	"""The inverse of sym_compress. Expands a flat array of numbers
	into a symmetric matrix with ncomp components using the given
	mapping which (or construct one using the given scheme)."""
	if which==None: which = compressed_order(mat.shape[axis], scheme=scheme)
	if ncomp==None: ncomp = np.max(which)+1
	m = np.rollaxis(mat, axis)
	shape = [ncomp,ncomp] + list(m.shape[1:])
	res = np.zeros(shape,dtype=mat.dtype)
	for i, w in enumerate(which):
		res[w[0],w[1]] = m[i]
		if w[0] != w[1]:
			res[w[1],w[0]] = m[i]
	return np.rollaxis(np.rollaxis(res, 1, axis), 0, axis)

def compressed_order(n, scheme=None):
	"""Surmise the order in which the unique elements of 
	a symmetric matrix are stored, based on the number of such
	elements. Three different schemes are supported. The best
	one is the "stable" scheme because it can be truncated
	without the entries changing their meaning. However,
	the default in healpy is "diag", so that is the default here too.

	stable:
		00
		00 11
		00 11 01
		00 11 01 22
		00 11 01 22 02
		00 11 01 22 02 12
		...

	diag:
		00
		00 11
		00 11 01
		00 11 22 01
		00 11 22 01 12
		00 11 22 01 12 02
		...

	row:
		00
		00 11
		00 01 11
		00 01 11 22
		00 01 02 11 22
		00 01 02 11 12 22
		..."""
	if scheme == None: scheme = "diag"
	# nfull = ncomp*(ncomp+1)/2 =>
	# ncomp = (-1+sqrt(1+8*nfull))/2
	ncomp = int(np.ceil((-1+(1+8*n)**0.5)/2))
	which = []
	if scheme == "stable":
		c = 0
		while len(which) < n:
			which.append([c,c])
			for i in range(min(c,n-len(which))):
				which.append([i,c])
			c += 1
	elif scheme == "row":
		m = ncomp
		for i in range(ncomp):
			for j in range(i, ncomp):
				if i != j:
					if m >= n: continue
					m += 1
				which.append([i,j])
	elif scheme == "diag":
		for d in range(ncomp):
			for i in range(0, ncomp-d):
				which.append([i,i+d])
	else:
		raise ValueError("Unknown scheme " + scheme)
	return which[:n]

def expand_inds(x, y):
	n = np.max(x)+1
	res = np.zeros((y.shape[0],n))
	res[:,x] = y
	return res

def scale_spectrum(a, direction):
	x = np.arange(a.shape[-1])
	a[...,1:] *= (2*np.pi/x[1:]/(x[1:]+1))**direction
	a[...,0] = 0
	return a

def read_spectrum(fname, inds=True, scale=True, expand=None):
	"""Read a power spectrum from disk and return a dense
	array cl[nspec,lmax+1]. Unless scale=False, the spectrum
	will be multiplied by 2pi/l/(l+1) when being read.
	Unless inds=False, the first column in the file is assumed
	to be the indices. If expand!=None, it can be one of the
	valid expansion schemes from compressed_order, and will
	cause the returned array to be cl[ncomp,ncomp,lmax+1]
	instaed."""
	a = np.loadtxt(fname).T
	if inds: a = expand_inds(np.array(a[0],dtype=int), a[1:])
	if scale: a = scale_spectrum(a, 1)
	if expand is not None: a = sym_expand(a, scheme=expand)
	return a
