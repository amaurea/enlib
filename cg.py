from __future__ import division, print_function
import numpy as np
# Implementation of preconditioned conjugate gradients. More general than
# the scipy version, in that it does not assume that it knows how to perform
# the dot product. This makes it possible to use this for distributed x
# vectors. It is pretty much a straight port of the fortran cg solver in
# the quiet pipeline.

def default_M(x):     return np.copy(x)
def default_dot(a,b): return a.dot(np.conj(b))

class CG:
	"""A simple Preconditioner Conjugate gradients solver. Solves
	the equation system Ax=b."""
	def __init__(self, A, b, x0=None, M=default_M, dot=default_dot):
		"""Initialize a solver for the system Ax=b, with a starting guess of x0 (0
		if not provided). Vectors b and x0 must provide addition and multiplication,
		as well as the .copy() method, such as provided by numpy arrays. The
		preconditioner is given by M. A and M must be functors acting on vectors
		and returning vectors. The dot product may be manually specified using the
		dot argument. This is useful for MPI-parallelization, for example."""
		# Init parameters
		self.A   = A
		self.b   = b
		self.M   = M
		self.dot = dot
		if x0 is None:
			self.x = b*0
			self.r = b
		else:
			self.x  = x0.copy()
			self.r  = b-self.A(self.x)
		# Internal work variables
		n = b.size
		z = self.M(self.r)
		self.rz  = self.dot(self.r, z)
		self.rz0 = float(self.rz)
		self.p   = z
		self.i   = 0
		self.err = np.inf
	def step(self):
		"""Take a single step in the iteration. Results in .x, .i
		and .err being updated. To solve the system, call step() in
		a loop until you are satisfied with the accuracy. The result
		can then be read off from .x."""
		Ap = self.A(self.p)
		alpha = self.rz/self.dot(self.p, Ap)
		self.x += alpha*self.p
		self.r -= alpha*Ap
		z       = self.M(self.r)
		next_rz = self.dot(self.r, z)
		self.err = next_rz/self.rz0
		beta = next_rz/self.rz
		self.rz = next_rz
		self.p  = z + beta*self.p
		self.i += 1
	def save(self, fname):
		"""Save the volatile internal parameters to hdf file fname. Useful
		for later restoring cg iteration"""
		import h5py
		with h5py.File(fname, "w") as hfile:
			for key in ["i","rz","rz0","x","r","p","err"]:
				hfile[key] = getattr(self, key)
	def load(self, fname):
		"""Load the volatile internal parameters from the hdf file fname.
		Useful for restoring a saved cg state, after first initializing the
		object normally."""
		import h5py
		with h5py.File(fname, "r") as hfile:
			for key in ["i","rz","rz0","x","r","p","err"]:
				setattr(self, key, hfile[key].value)

class BCG:
	"""A simple Preconditioner Biconjugate gradients stabilized solver. Solves
	the equation system Ax=b, where A is a (possibly asymmetric) matrix."""
	def __init__(self, A, b, x0=None, M=default_M, M2=default_M, dot=default_dot):
		"""Initialize a solver for the system Ax=b, with a starting guess of x0 (0
		if not provided). Vectors b and x0 must provide addition and multiplication,
		as well as the .copy() method, such as provided by numpy arrays. The
		preconditioner is given by M. If M2 is specified, then M=M1 M2. where
		M1 and M2 are the left and right preconditioners respectively. A, M and M2
		must all be functors acting on vectors and returning vectors. The dot
		product may be manually specified using the dot argument. This is useful
		for MPI-parallelization, for example."""
		# Init parameters
		self.A = A
		self.b = b
		self.M = M
		self.M2 = M2
		self.dot = dot
		self.bnorm = self.dot(b,b)
		if x0 is None:
			self.x = b*0
		else:
			self.x   = x0.copy()
		# Internal work variables
		n = b.size
		self.r  = b-self.A(self.x)
		self.rh = self.r.copy()
		self.rho, self.alpha, self.omega = 1.0, 1.0, 1.0
		self.AMp, self.p = self.x*0, self.x*0
		self.d = 4
		self.arz = []
		self.err_true = np.inf
		self.err = np.inf
		self.i = 0
	def step(self):
		"""Take a single step in the iteration. Results in .x, .i
		and .err being updated. To solve the system, call step() in
		a loop until you are satisfied with the accuracy. The result
		can then be read off from .x."""
		rho        = self.dot(self.rh, self.r)
		beta       = (rho/self.rho)*(self.alpha/self.omega)
		self.rho   = rho
		self.p     = self.r + beta*(self.p - self.omega*self.AMp)
		Mp         = self.M(self.p)
		self.AMp   = self.A(Mp)
		self.alpha = rho/self.dot(self.rh, self.AMp)
		s          = self.r - self.alpha*self.AMp
		Ms         = self.M(s)
		AMs        = self.A(Ms)
		self.omega = self.dot(self.M2(AMs),self.M2(s))/self.dot(self.M2(AMs),self.M2(AMs))
		self.x    += self.alpha*Mp + self.omega*Ms
		self.r     = s - self.omega*AMs
		self.err   = self.dot(self.r,self.r)/self.bnorm
		self.i    += 1

def cg_test():
	def A(x): return np.array([[4,1],[1,3]],dtype=float).dot(x)
	b = np.array([1.,2])
	cg = CG(A, b, x0=np.array([2.,1.]))
	while cg.err > 1e-4:
		cg.step()
		print(cg.i, cg.err, cg.x)
def bcg_test():
	def A(x): return np.array([[4,2],[1,3]],dtype=float).dot(x)
	b = np.array([1.,2])
	cg = BCG(A, b, x0=np.array([2.,1.]))
	while cg.err > 1e-4:
		cg.step()
		print(cg.i, cg.err, cg.x)
