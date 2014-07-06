import numpy as np
from fortran import array_ops
from enlib import utils

def get_dtype_fun(funcs, dtype):
	if dtype.char in funcs:
		return funcs[dtype.char]
	else:
		raise NotImplementedError("Only dtypes " + ", ".join([key for key in funcs]) + " implemented")

dtype_map = { np.float32: "32", np.float64: "64", np.complex64: "c64", np.complex128: "c128" }
# FIXME: These functions all operate on the transposed matrices, i.e. matmul(A,B) really does
# matmul(A.T,B.T).T = matmul(B,A).
def gen_wrap2(vec2mat=False, **funcs):
	def f(A,b,axes=[-2,-1]):
		axes = [i if i >= 0 else A.ndim+i for i in axes]
		b  = b.copy()
		Af = utils.partial_flatten(A,axes)
		bf = utils.partial_flatten(b,axes[:len(axes)-(A.ndim-b.ndim)])
		if vec2mat and bf.ndim == 2: bf = bf.reshape(bf.shape[:-1]+(1,bf.shape[-1]))
		b2 = np.ascontiguousarray(bf)
		assert A.dtype == b.dtype
		fun = get_dtype_fun(funcs, A.dtype)
		fun(Af.T, b2.T)
		if bf is not b2: bf[...] = b2[...]
		return b
	return f

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
matmul = gen_wrap2(vec2mat=True,
		f=array_ops.matmul_32,
		d=array_ops.matmul_64,
		F=array_ops.matmul_c64,
		D=array_ops.matmul_c128)
solve_multi = gen_wrap2(
		f=array_ops.solve_multi_32,
		d=array_ops.solve_multi_64,
		F=array_ops.solve_multi_c64,
		D=array_ops.solve_multi_c128)
solve_masked = gen_wrap2(
		f=array_ops.solve_masked_32,
		d=array_ops.solve_masked_64,
		F=array_ops.solve_masked_c64,
		D=array_ops.solve_masked_c128)
condition_number_multi = gen_wrap1(
		f=array_ops.condition_number_multi_32,
		d=array_ops.condition_number_multi_64)
eigpow = gen_mat2mat(
		f=array_ops.eigpow_32,
		d=array_ops.eigpow_64)

