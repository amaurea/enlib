import numpy as np
from .. import utils
import fortran_32, fortran_64, fortran_c64, fortran_c128

def get_core(dtype):
	if   dtype == np.float32:    return fortran_32.array_ops
	elif dtype == np.float64:    return fortran_64.array_ops
	elif dtype == np.complex64:  return fortran_c64.array_ops
	elif dtype == np.complex128: return fortran_c128.array_ops
	raise ValueError("Unsupported data type: %s" % str(dtype))

def get_f(c):
	if   c == 'f': return fortran_32.array_ops
	elif c == 'd': return fortran_64.array_ops
	elif c == 'F': return fortran_c64.array_ops
	elif c == 'D': return fortran_c128.array_ops
	raise ValueError("Unrecognized data char '%d'" % c)

def get_funcs(name, chars='fdFD'):
	return {c: getattr(get_f(c),name) for c in chars}

def get_dtype_fun(funcs, dtype):
	if dtype.char in funcs:
		return funcs[dtype.char]
	else:
		raise NotImplementedError("Only dtypes " + ", ".join([key for key in funcs]) + " implemented")

#def parallel_flatten(a, axes, comm=None):
#	af = utils.partial_flatten(a, axes)
#	if False and comm:
#		i,n,N = comm.rank,comm.size,af.shape[0]
#		af = af[i*N/n:(i+1)*N/n]
#	return af
#
#def parallel_expand(af, shape, axes, comm=None):
#	if True or comm is None or comm.size == 1:
#		return utils.partial_expand(af, shape, axes)
#	i,n = comm.rank, comm.size
#	N = np.product([s for i,s in enumerate(shape) if i not in axes])
#	aftot = np.zeros((N,)+af.shape[1:],af.dtype)
#	aftot[i*N/n:(i+1)*N/n] = af
#	out = aftot.copy()
#	comm.Allreduce(aftot, out)
#	return utils.partial_expand(out, shape, axes)

def wrap_mm_m(vec2mat=False, **funcs):
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
		if mustadd: Bf = utils.addaxes(Bf, [1])
		Bf = np.ascontiguousarray(Bf)
		assert A.dtype == B.dtype
		fun = get_dtype_fun(funcs, A.dtype)
		fun(Af.T, Bf.T)
		if mustadd: Bf = utils.delaxes(Bf, [1])
		B[...] = utils.partial_expand(Bf, B.shape, bax)
		return B
	return f

def wrap_m_m(**funcs):
	def f(A,*args,**kwargs):
		axes = kwargs["axes"] if "axes" in kwargs else [-2,-1]
		axes = [i if i >= 0 else A.ndim+i for i in axes]
		A  = A.copy()
		Af = np.ascontiguousarray(utils.partial_flatten(A,axes))
		fun = get_dtype_fun(funcs, A.dtype)
		fun(Af.T, *args)
		return utils.partial_expand(Af, A.shape, axes)
	return f

# FIXME: These functions all operate on the transposed matrices, i.e. matmul(A,B) really does
# matmul(A.T,B.T).T = matmul(B,A).
#def gen_wrap2(vec2mat=False, **funcs):
#	def f(A,b,axes=[-2,-1]):
#		axes = [i if i >= 0 else A.ndim+i for i in axes]
#		b  = b.copy()
#		Af = utils.partial_flatten(A,axes)
#		bf = utils.partial_flatten(b,axes[:len(axes)-(A.ndim-b.ndim)])
#		if vec2mat and bf.ndim == 2: bf = bf.reshape(bf.shape[:-1]+(1,bf.shape[-1]))
#		b2 = np.ascontiguousarray(bf)
#		assert A.dtype == b.dtype
#		fun = get_dtype_fun(funcs, A.dtype)
#		fun(Af.T, b2.T)
#		if bf is not b2: bf[...] = b2[...]
#		if b is not bf:
#			print b.shape, bf.shape
#			return utils.partial_expand(bf, b.shape, axes[:len(axes)-(A.ndim-b.ndim)])
#		return b
#	return f

def gen_wrap1(**funcs):
	def f(A,axes=[-2,-1]):
		axes = [i if i >= 0 else A.ndim+i for i in axes]
		b  = utils.moveaxes(A,axes,[0,1])[0,0].copy()
		Af = utils.partial_flatten(A,axes)
		bf = b.reshape(-1)
		assert A.dtype == b.dtype
		fun = get_dtype_fun(funcs, A.dtype)
		fun(Af.T, bf.T)
		return b
	return f

def gen_mat2mat(**funcs):
	def f(A,*args,**kwargs):
		axes = kwargs["axes"] if "axes" in kwargs else [-2,-1]
		axes = [i if i >= 0 else A.ndim+i for i in axes]
		A  = A.copy()
		Af = utils.partial_flatten(A,axes)
		A2 = np.ascontiguousarray(Af)
		fun = get_dtype_fun(funcs, A.dtype)
		fun(A2.T, *args)
		if Af is not A2: Af[...] = A2[...]
		return A
	return f

# Our wrapped functions
matmul_sym = wrap_mm_m(vec2mat=True, **get_funcs("matmul_multi_sym"))
solve_multi = wrap_mm_m(**get_funcs("solve_multi"))
solve_masked = wrap_mm_m(**get_funcs("solve_masked"))
#condition_number_multi = gen_wrap1(**get_funcs("condition_number_multi"))
eigflip = wrap_m_m(**get_funcs("eigflip"))

def measure_cov(d, delay=0):
	cov = np.empty([d.shape[0],d.shape[0]],dtype=d.real.dtype)
	fun = get_dtype_fun(get_funcs("measure_cov"), d.dtype)
	fun(d.T,cov.T,delay)
	return cov

def ang2rect(a):
	core = get_core(a.dtype)
	res = np.zeros([len(a),3],dtype=a.dtype)
	core.ang2rect(a.T,res.T)
	return res

def eigpow(A, pow, axes=[-2,-1], lim=None, lim0=None, copy=True, fallback="eigkill"):
	core = get_core(A.dtype)
	if lim  is None: lim  = 1e-6
	if lim0 is None: lim0 = np.finfo(A.dtype).tiny**0.2
	if copy: A = A.copy()
	with utils.flatview(A, axes=axes) as Af:
		if   fallback == "eigkill":
			core.eigpow(Af.T, pow, lim, lim0)
		elif fallback == "scalar":
			core.eigpow_scalar_fallback(Af.T, pow, lim, lim0)
		else:
			raise ValueError("Unknown fallback in eigpow: '%s'" % str(fallback))
	return A

def condition_number_multi(A, axes=[-2,-1]):
	core = get_core(A.dtype)
	inds = [slice(None) for i in range(A.ndim)]
	for ax in axes: inds[ax] = 0
	b = A.real[tuple(inds)].copy()
	with utils.flatview(A, axes=axes) as Af:
		with utils.flatview(b, axes=axes) as bf:
			core.condition_number_multi(Af.T,bf.T)
	return b

def svdpow(A, pow, axes=[-2,-1], lim=None, lim0=None, copy=True):
	core = get_core(A.dtype)
	if lim  is None: lim  = 1e-6
	if lim0 is None: lim0 = np.finfo(A.dtype).tiny**0.2
	if copy: A = A.copy()
	with utils.flatview(A, axes=axes) as Af:
		core.svdpow(Af.T, pow, lim, lim0)
	return A

def matmul(A,B,axes=[-2,-1]):
	# Massage input arrays. This should be factored out,
	# as it is common for many functions
	axes = [i if i >= 0 else A.ndim+i for i in axes]
	bax  = axes[:len(axes)-(A.ndim-B.ndim)]
	Af = utils.partial_flatten(A,axes)
	Bf = utils.partial_flatten(B,bax)
	mustadd = Bf.ndim == 2
	if mustadd: Bf = utils.addaxes(Bf, [1])
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
	if mustadd: Xf = utils.delaxes(Xf, [1])
	X = utils.partial_expand(Xf, B.shape, bax)
	return X

# This one might belong in a different module.
# In general array_ops needs to be cleaned up.
def find_contours(imap, vals, omap=None):
	core = get_core(imap.dtype)
	if omap is None:
		omap = imap.astype(np.int32)*0
	core.find_contours(imap.T, vals, omap.T)
	return omap
