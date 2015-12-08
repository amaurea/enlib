"""This module deals with the representation of degrees of freedom for
solving equations etc. In python these are often most naturally described
as several distinct objects, each with some of the degrees of freedom,
but solvers need to see these are a single vector of elements. The
classes here implement the necessary mapping these representations."""
import numpy as np

class Arg:
	def __init__(self, shape=None, mask=None, default=None, array=None, distributed=False):
		"""array is an alias for default"""
		self.distributed = distributed
		self.mask        = mask
		shapes = []
		if shape is not None:
			shapes.append(shape)
		if mask is not None:
			shapes.append(mask.shape)
		if array is not None: default = array
		if default is not None:
			default = np.asanyarray(default)
			if default.ndim > 0:
				shapes.append(default.shape)
		assert len(shapes) > 0, "At least one of shape, mask, default is needed"
		for s in shapes[1:]:
			assert s == shapes[0], "Incompatible shapes in DOF: " + ", ".join(shapes)
		self.default     = default
		self.shape = shapes[0]
		self.size  = np.product(self.shape)
		self.n     = self.size if mask is None else np.sum(mask)
	def __repr__(self):
		res = "Arg(shape="+str(self.shape)+",n="+str(self.n)
		if self.mask is not None: res += ",mask=..."
		if self.default is not None: res += ",default=..."
		if self.distributed: res += ",distributed=True"
		return res + ")"

class DOF:
	"""DOF represents a mapping between numbers in a set of arrays and a
	set of degrees of freedom, represented as a plain 1-dimensional array. This
	is useful for solving equation systems where elements of various arrays
	enter as degrees of freedom."""
	def __init__(self, *args, **kwargs):
		"""DOF(info, info, info, ...., [comm=comom], where info is a DOFArg"""
		self.info  = args
		comm = kwargs["comm"] if "comm" in kwargs else None
		n = 0
		r = []
		for info in self.info:
			r.append([n,n+info.n])
			n += info.n
		self.n = n
		self.r = np.asanyarray(r)

		# Set up our communicator if necessary
		comm_needed = any([info.distributed for info in self.info])
		if comm_needed and comm is None:
			from enlib import mpi
			comm = mpi.COMM_WORLD
		self.comm = comm
	def zip(self, *args):
		"""x = DOF.zip(arr1, arr2, ...)."""
		args  = [np.asanyarray(a) for a in args]
		dtype = np.result_type(*tuple([a.dtype for a in args]))
		res = np.empty(self.n, dtype)
		for info, r, arg in zip(self.info, self.r, args):
			targ = res[r[0]:r[1]]
			targ[...] = arg[info.mask] if info.mask is not None else arg.reshape(-1)
		return res
	def unzip(self, x):
		"""arr1, arr2, ... = DOF.unzip(x)"""
		res = []
		for info, r in zip(self.info, self.r):
			source = x[r[0]:r[1]]
			if info.mask is not None:
				a = info.mask.astype(x.dtype,copy=True)
				if info.default is not None:
					a[...] = info.default
				a[info.mask] = source
				res.append(a)
			else:
				if info.default is not None:
					a = info.default.copy()
					a[...] = source.reshape(a.shape)
				else:
					a = source.copy().reshape(info.shape)
				res.append(a)
		return tuple(res)
	def dot(self, x, y):
		"""Dot product of arrays x and y. Defined as sum(x*y) here,
		except that distributed arrays are properly reduced."""
		res = [0,0]
		for info, r in zip(self.info, self.r):
			res[info.distributed] += np.sum(x[r[0]:r[1]]*y[r[0]:r[1]])
		if self.comm is None:
			return np.sum(res)
		else:
			return res[0] + self.comm.allreduce(res[1])
	def reduce(self, x):
		"""Sum up the shared parts of x as passed from different processes, leaving the
		unshared parts untourched"""
		res = x.copy()
		for info, r in zip(self.info, self.r):
			if info.distributed and self.comm is not None:
				self.comm.Allreduce(x[r[0]:r[1]], res[r[0]:r[1]])
		return res
	def __repr__(self):
		return "DOF("+",".join([str(info) for info in self.info])+"){n="+str(self.n)+"}"

class OldDOF:
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
			if type(a) is tuple:
				self.masks.append(None)
				self.shapes.append(a)
				m = np.prod(a)
			else:
				a = np.atleast_1d(a)
				if a.dtype == bool:
					self.masks.append(a)
					self.shapes.append(a.shape)
					m = np.sum(a)
				else:
					self.masks.append(a.astype(bool)+True)
					self.shapes.append(a.shape)
					m = np.prod(a.shape)
			self.r.append([n,n+m])
			self.sizes.append(m)
			self.dist.append(distributed)
			n += m
		self.r = np.asanyarray(self.r)
		self.n = n

		# Set up our communicator if necessary
		if any(self.dist) and comm is None:
			from mpi4py import MPI
			comm = MPI.COMM_WORLD
		self.comm = comm

	def zip(self, *args):
		"""x = flatterner.flatten(arr1, arr2, ...)."""
		args = [np.asanyarray(a) for a in args]
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
			res[dist] += np.sum(x[r[0]:r[1]]*y[r[0]:r[1]])
		if self.comm is None:
			return np.sum(res)
		else:
			return res[0] + self.comm.allreduce(res[1])
