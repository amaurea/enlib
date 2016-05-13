import numpy as np
from enlib import utils

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
	a = np.array(a)
	l = np.arange(a.shape[-1])
	a[...,1:] *= (2*np.pi/l[1:]/(l[1:]+1))**direction
	a[...,0] = 0
	return a

def scale_lensing_spectrum(a, direction):
	a = np.array(a)
	l = np.arange(a.shape[-1])
	a[...,1:] /= (l[1:]**4*2.726e6**2)**direction
	a[...,0] = 0
	return a

def read_spectrum(fname, inds=True, scale=True, expand="diag", ncol=None, ncomp=None):
	"""Read a power spectrum from disk and return a dense
	array cl[nspec,lmax+1]. Unless scale=False, the spectrum
	will be multiplied by 2pi/l/(l+1) when being read.
	Unless inds=False, the first column in the file is assumed
	to be the indices. If expand!=None, it can be one of the
	valid expansion schemes from compressed_order, and will
	cause the returned array to be cl[ncomp,ncomp,lmax+1]
	instaed."""
	a = np.atleast_2d(np.loadtxt(fname).T)
	if inds: a = expand_inds(np.array(a[0],dtype=int), a[1:])
	if scale: a = scale_spectrum(a, 1)
	if ncol: a = a[:ncol]
	if expand is not None: a = sym_expand(a, scheme=expand, ncomp=ncomp)
	return a

def read_lensing_spectrum(fname, coloff=0, inds=True, scale=True, expand="diag"):
	a = read_spectrum(fname, inds=inds, scale=False, expand=None)[coloff]
	if scale: a = scale_lensing_spectrum(a, 1)
	if expand is not None: a = a[None,None]
	return a

def read_camb_scalar(fname, inds=True, scale=True, expand=True, ncmb=3):
	if expand: expand = "diag"
	ps_cmb  = read_spectrum(fname, inds=inds, scale=scale, expand=expand, ncol=ncmb, ncomp=3)
	ps_lens = read_lensing_spectrum(fname, inds=inds, scale=scale, expand=expand, coloff=ncmb)
	return ps_cmb, ps_lens

def write_spectrum(fname, spec, inds=True, scale=True, expand="diag"):
	if scale: spec = scale_spectrum(spec, -1)
	if expand is not None: spec = sym_compress(spec, scheme=expand)
	if inds: spec = np.concatenate([np.arange(spec.shape[-1])[None],spec],0)
	np.savetxt(fname, spec.T, fmt="%15.7e")

def spec2corr(spec, pos, iscos=False, symmetric=True):
	"""Compute the correlation function sum(2l+1)/4pi Cl Pl(cos(theta))
	corresponding to the given power spectrum at the given positions."""
	spec = np.asarray(spec)
	pos  = np.asarray(pos)
	if not iscos: pos = np.cos(pos)
	if symmetric: fspec = sym_compress(spec)
	else: fspec = spec.reshape(-1,spec.shape[-1])
	l = np.arange(spec.shape[-1])
	weight = (2*l+1)/(4*np.pi)
	res = np.zeros(fspec.shape[:1]+pos.shape)
	for i, cl in enumerate(fspec):
		res[i] = np.polynomial.legendre.legval(pos, weight*cl)
	if symmetric: res = sym_expand(res)
	return res
