import numpy as np
from fortran import array_ops
from enlib import utils

def gen_wrap2(fun32, fun64):
	def f(A,b,axes=[-2,-1]):
		axes = [i if i >= 0 else A.ndim+i for i in axes]
		b  = b.copy()
		Af = utils.partial_flatten(A,axes)
		bf = utils.partial_flatten(b,axes[:A.ndim-b.ndim])
		b2 = np.ascontiguousarray(bf)
		assert A.dtype == b.dtype
		if A.dtype == np.float32:
			fun32(Af.T, b2.T)
		elif A.dtype == np.float64:
			fun64(Af.T, b2.T)
		else:
			raise NotImplementedError("Only 32-bit and 64-bit floats supported")
		if bf is not b2: bf[...] = b2[...]
		return b
	return f

def gen_wrap1(fun32, fun64):
	def f(A,axes=[-2,-1]):
		print axes
		axes = [i if i >= 0 else A.ndim+i for i in axes]
		b  = utils.moveaxes(A,axes,[0,1])[0,0].copy()
		Af = utils.partial_flatten(A,axes)
		bf = b.reshape(-1)
		assert A.dtype == b.dtype
		if A.dtype == np.float32:
			fun32(Af.T, bf.T)
		elif A.dtype == np.float64:
			fun64(Af.T, bf.T)
		else:
			raise NotImplementedError("Only 32-bit and 64-bit floats supported")
		return b
	return f

solve_multi   = gen_wrap2(array_ops.solve_multi_32, array_ops.solve_multi_64)
solve_masked  = gen_wrap2(array_ops.solve_masked_32, array_ops.solve_masked_64)
condition_number_multi  = gen_wrap1(array_ops.condition_number_multi_32, array_ops.condition_number_multi_64)
