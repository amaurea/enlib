import numpy as np
from fortran import array_ops
from enlib import utils

def get_dtype_fun(funcs, dtype):
	if dtype.char in funcs:
		return funcs[dtype.char]
	else:
		raise NotImplementedError("Only dtypes " + ", ".join([key for key in funcs]) + " implemented")

dtype_map = { np.float32: "32", np.float64: "64", np.complex64: "c64", np.complex128: "c128" }

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
matmul = wrap_mm_m(vec2mat=True,
		f=array_ops.matmul_32,
		d=array_ops.matmul_64,
		F=array_ops.matmul_c64,
		D=array_ops.matmul_c128)
solve_multi = wrap_mm_m(
		f=array_ops.solve_multi_32,
		d=array_ops.solve_multi_64,
		F=array_ops.solve_multi_c64,
		D=array_ops.solve_multi_c128)
solve_masked = wrap_mm_m(
		f=array_ops.solve_masked_32,
		d=array_ops.solve_masked_64,
		F=array_ops.solve_masked_c64,
		D=array_ops.solve_masked_c128)
condition_number_multi = gen_wrap1(
		f=array_ops.condition_number_multi_32,
		d=array_ops.condition_number_multi_64)
eigpow = wrap_m_m(
		f=array_ops.eigpow_32,
		d=array_ops.eigpow_64)

