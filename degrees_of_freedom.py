"""This module deals with the representation of degrees of freedom for
solving equations etc. In python these are often most naturally described
as several distinct objects, each with some of the degrees of freedom,
but solvers need to see these are a single vector of elements. The
classes here implement the necessary mapping these representations."""
import numpy as np

class DOF:
	"""DOF represents a mapping between numbers in a set of arrays and a
	set of degrees of freedom, represented as a plain 1-dimensional array. This
	is useful for solving equation systems where elements of various arrays
	enter as degrees of freedom."""
	def __init__(self, *args, **kwargs):
		"""DOF(info, info, info, ..., [comm=comm]), where info is either:
			1. shape, where shape is a valid array shape as accepted by np.zeros.
			2. boolean array of the same shape etc. as the one the flatterner will be
				used with later. This form has the advantage of resulting in an .expand()
				which uses the correct array subclass. This array also acts as a mask:
				False elements will be ignored when flattening.
			3. A dict {desc:info}, where info is of type 1 or 2 above, and desc
				is a string "shared" or "s" for a non-distributed array and
				"dist" or "d" for a distributed array. Arrays markes as distributed
				will be MPI reduced."""
		self.masks  = []
		self.shapes = []
		self.sizes  = []
		self.r      = []
		self.dist   = []
		comm = kwargs["comm"] if "comm" in kwargs else None
		n = 0
		for a in args:
			distributed = False
			if type(a) is dict:
				k = a.keys()[0]
				distributed = k in ["dist", "distributed", "d"]
				a = a[k]
			a = np.atleast_1d(a)
			if a.dtype == bool:
				self.masks.append(a)
				self.shapes.append(a.shape)
				m = np.sum(a)
			else:
				assert a.ndim == 1
				self.masks.append(None)
				self.shapes.append(tuple(a))
				m = np.prod(a)
			self.r.append([n,n+m])
			self.sizes.append(m)
			self.dist.append(distributed)
			n += m
		self.r = np.asarray(self.r)
		self.n = n

		# Set up our communicator if necessary
		if any(self.dist) and comm is None:
			from mpi4py import MPI
			comm = MPI.COMM_WORLD
		self.comm = comm

	def zip(self, *args):
		"""x = flatterner.flatten(arr1, arr2, ...)."""
		args = [np.asarray(a) for a in args]
		res = np.empty(self.n, args[0].dtype)
		for r, mask, arg in zip(self.r, self.masks, args):
			targ = res[r[0]:r[1]]
			targ[...] = arg[mask] if mask is not None else arg.reshape(-1)
		return res
	def unzip(self, x):
		"""arr1, arr2, ... = flattener.expand(x)"""
		res = []
		for r, mask, shape in zip(self.r, self.masks, self.shapes):
			source = x[r[0]:r[1]]
			if mask is not None:
				a = mask.astype(x.dtype)
				a[mask] = source
				res.append(a)
			else:
				res.append(source.copy().reshape(shape))
		return tuple(res)
	def dot(self, x, y):
		"""Dot product of arrays x and y. Defined as sum(x*y) here,
		except that distributed arrays are properly reduced."""
		res = [0,0]
		for r, dist in zip(self.r, self.dist):
			res[dist] += np.sum(x[r[0]:r[1]]*x[r[0]:r[1]])
		if self.comm is None:
			return np.sum(res)
		else:
			return res[0] + self.comm.allreduce(res[1])
