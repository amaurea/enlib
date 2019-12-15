from __future__ import division, print_function
import numpy as np
from .. import utils
from . import fortran_32, fortran_64, fortran_c64, fortran_c128

def get_core(dtype):
	if   dtype == np.float32:    return fortran_32.array_ops
	elif dtype == np.float64:    return fortran_64.array_ops
	elif dtype == np.complex64:  return fortran_c64.array_ops
	elif dtype == np.complex128: return fortran_c128.array_ops
	raise ValueError("Unsupported data type: %s" % str(dtype))

def measure_cov(d, delay=0):
	core = get_core(d.dtype)
	cov  = np.empty([d.shape[0],d.shape[0]],dtype=d.real.dtype)
	core.measure_cov(d.T,cov.T,delay)
	return cov

def ang2rect(a):
	core = get_core(a.dtype)
	res = np.zeros([len(a),3],dtype=a.dtype)
	core.ang2rect(a.T,res.T)
	return res

def eigpow(A, pow, axes=[-2,-1], lim=None, lim0=None, copy=True, fallback="eigkill"):
	core = get_core(A.dtype)
	if lim  is None: lim  = 1e-6
	if lim0 is None: lim0 = np.finfo(A.dtype).tiny**0.5
	if copy: A = A.copy()
	with utils.flatview(A, axes=axes) as Af:
		if   fallback == "eigkill":
			core.eigpow(Af.T, pow, lim, lim0)
		elif fallback == "scalar":
			core.eigpow_scalar_fallback(Af.T, pow, lim, lim0)
		else:
			raise ValueError("Unknown fallback in eigpow: '%s'" % str(fallback))
	return A

def eigflip(A, axes=[-2,-1], inplace=False):
	core = get_core(A.dtype)
	if not inplace: A = np.array(A)
	with utils.flatview(A, axes=axes) as Af:
		core.eigflip(Af.T)
	return A

def condition_number_multi(A, axes=[-2,-1]):
	core = get_core(A.dtype)
	inds = [slice(None) for i in range(A.ndim)]
	for ax in axes: inds[ax] = 0
	b = A.real[tuple(inds)].copy()
	with utils.flatview(A, axes=axes, mode="rc") as Af:
		with utils.flatview(b, axes=axes, mode="rwc") as bf:
			core.condition_number_multi(Af.T,bf.T)
	return b

def matmul(A, B, axes=[-2,-1]):
	# Massage input arrays. This should be factored out,
	# as it is common for many functions
	axes = [i if i >= 0 else A.ndim+i for i in axes]
	bax  = axes[:len(axes)-(A.ndim-B.ndim)]
	Af = utils.partial_flatten(A,axes)
	Bf = utils.partial_flatten(B,bax)
	mustadd = Bf.ndim == 2
	if mustadd: Bf = Bf[:,None,:]
	Bf = np.ascontiguousarray(Bf)
	if A.dtype != B.dtype:
		dtype = np.result_type(A.dtype,B.dtype)
		Af = Af.astype(dtype,copy=False)
		Bf = Bf.astype(dtype,copy=False)
	# Compute the shape of the output array
	Xf = np.empty((Bf.shape[0],Bf.shape[1],Af.shape[1]),dtype=Bf.dtype)
	# Actually perform the operation
	core = get_core(Bf.dtype)
	core.matmul_multi(Af.T, Bf.T, Xf.T)
	# Unwrangle
	if mustadd: Xf = Xf[:,0,:]
	X = utils.partial_expand(Xf, B.shape, bax)
	return X

# This one might belong in a different module.
def find_contours(imap, vals, omap=None):
	core = get_core(imap.dtype)
	if omap is None:
		omap = imap.astype(np.int32)*0
	core.find_contours(imap.T, vals, omap.T)
	return omap

def maxbin(map, inds, vals):
	core = get_core(imap.dtype)
	core.maxbin(map, inds, vals)

def wrap_mm_m(name, vec2mat=False):
	"""Wrap a fortran subroutine which takes (n,n,m),(n,k,m) and overwrites
	its second argument to a python function where the "n" axes can be
	at arbitrary locations, specified by the axes argument, and where
	the arrays can be arbitrary-dimensional. These are all flattened
	internally. If vec2mat is specified, the second argument will have
	a dummy axis added internally if needed."""
	def f(A,B,axes=[-2,-1]):
		axes = [i if i >= 0 else A.ndim+i for i in axes]
		bax  = axes[:len(axes)-(A.ndim-B.ndim)]
		B  = B.copy()
		Af = utils.partial_flatten(A,axes)
		Bf = utils.partial_flatten(B,bax)
		mustadd = vec2mat and Bf.ndim == 2
		if mustadd: Bf = Bf[:,None,:]
		Bf = np.ascontiguousarray(Bf)
		assert A.dtype == B.dtype
		core = get_core(A.dtype)
		fun  = getattr(core, name)
		fun(Af.T, Bf.T)
		if mustadd: Bf = Bf[:,0,:]
		B[...] = utils.partial_expand(Bf, B.shape, bax)
		return B
	return f

matmul_sym   = wrap_mm_m("matmul_multi_sym", vec2mat=True)
solve_multi  = wrap_mm_m("solve_multi")
solve_masked = wrap_mm_m("solve_masked")

