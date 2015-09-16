"""This module provides functions for flattening (zipping) and unflattening (unzipping)
arrays or objects into one-dimensional numpy arrays. It is similar to pickle, but
faster and less general. It is intended as a more general replacement of degrees_of_freedom.py."""
import numpy as np

class SingleZipper:
	"""General interface to sinle-object zippers. Provides zip, unzip,
	n, and shared. The latter is only useful in distributed systems, and
	tells us whether the data in this object are reduntantly shared
	across multiple tasks, or whether each task contains its own subset
	of the data. It can usually be ignored."""
	def __init__(self, shared=True, comm=None):
		self.shared = shared
		self.comm = getcomm(comm, shared)
	def zip(self, a):
		"""Zips object a into a 1d numpy array."""
		raise NotImplementedError
	def unzip(self, x):
		"""Unzips 1d numpy array x into an object a."""
		raise NotImplementedError
	def sum(self, x):
		"""Sums 1d numpy array x, taking into account mpi distribution."""
		res = np.sum(x)
		if not self.shared: res = self.comm.allreduce(res)
		return res
	def dot(self, x, y): return self.sum(x*y)
	n = 0
	shared = True

def getcomm(comm, shared):
	if comm is not None or shared: return comm
	from mpi4py import MPI
	return MPI.COMM_WORLD

class ArrayZipper(SingleZipper):
	"""Zipper for numpy arrays and derived classes."""
	def __init__(self, template, mask=None, shared=True, comm=None):
		SingleZipper.__init__(self, shared, comm)
		self.template, self.mask = template, mask
		self.n = np.size(template) if mask is None else np.sum(mask)
	def zip(self, a):
		return a.reshape(-1) if self.mask is None else a[self.mask]
	def unzip(self, x):
		if self.mask is None:
			self.template[...] = x.reshape(self.template.shape)
		else:
			self.template[self.mask] = x
		return self.template

class MultiZipper:
	"""Meta-zipper"""
	def __init__(self, zippers, comm=None):
		self.zippers = zippers
		cum = np.concatenate([[0],np.cumsum([z.n for z in zippers])])
		self.n = cum[-1]
		self.bins = np.array([cum[:-1],cum[1:]]).T
		self.allshared = np.all([z.shared for z in zippers])
		self.comm = getcomm(comm, self.allshared)
		self.shared = True
	def zip(self, args):
		return np.concatenate([z.zip(a) for z,a in zip(self.zippers, args)])
	def unzip(self, x):
		return tuple([z.unzip(x[b[0]:b[1]]) for z,b in zip(self.zippers, self.bins)])
	def sum(self, x):
		if self.allshared: return np.sum(x)
		s = [0,0]
		for z,b in zip(self.zippers, self.bins):
			s[z.shared] += np.sum(x[b[0]:b[1]])
		return self.comm.allreduce(s[0]) + s[1]
	def dot(self, x, y): return self.sum(x*y)
