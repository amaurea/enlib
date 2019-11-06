from __future__ import division, print_function
import numpy as np
from . import enmap, array_ops, cg
from .degrees_of_freedom import DOF, Arg

def mul(mat, vec, axes):
	return array_ops.matmul(mat.astype(vec.dtype),vec,axes=axes)
def pow(mat, exp, axes): return array_ops.eigpow(mat,exp,axes=axes)

class FieldSampler:
	"""Draws samples from P(s|d,S,N), under the model d <- N(0,S+N).
	iS and iN must be (...,ncomp,ncomp,ny,nx), while d is
	(...,ncomp,ny,nx). "i" indicates the inverse here. The axes
	argument can be used to change which axes are regarded as the
	component directions. The "mode" argument describes the basis
	iS and iN are expesse in. The first letter corresponds to iS,
	the second to iN. "h" means harmonic space, and "p" means
	pixel space. Capitalized letters indicates which space to use
	for the internal preconditioner.

	The actual sampling is done using the .sample() method.
	To get predictable values, use np.random.seed(). The class
	can also be used to wiener-filter, using the .wiener() method."""
	def __init__(self, iS, iN, data=None, mode="Hp", axes=[-4,-3]):
		self.mode=mode
		self.imats = [iS, iN]
		self.hmats = [pow(imat, 0.5, axes=axes) for imat in self.imats]
		self.axes  = axes
		self.trf = {
				"p":(lambda x:x, lambda x:x),
				"h":(enmap.map2harm, enmap.harm2map)
			}
		# Preconditioner
		self.pind = [c.isupper() for c in mode+"F"].index(True)%2
		if mode.lower()[0] == mode.lower()[1]:
			self.prec = pow(np.sum(self.imats,0),-1,axes=axes)
		else:
			white= np.mean(np.mean(self.imats[1-self.pind],-1),-1)
			self.prec = pow(self.imats[self.pind]+white.T[None,None].T,-1,axes=axes)
		self.x, self.b = None,None
		self.set_data(data)
	def M(self, vec):
		return self.Tmul(self.mode[self.pind], self.prec, vec)
	def A(self, vec):
		mode = self.mode.lower()
		if mode[0] == mode[1]:
			return self.Tmul(mode[0], np.sum(self.imats,0), vec)
		else:
			Av1 = self.Tmul(mode[0], self.imats[0], vec)
			Av2 = self.Tmul(mode[1], self.imats[1], vec)
			return Av1 + Av2
	def set_data(self, data):
		"""Updates the constraining data d which determines the
		distribution P(s|d,S,N) from which this class draws samples."""
		self.d = data
		self.Nd = self.Tmul(self.mode[1], self.imats[1], data) if data is not None else None
	def solve(self, b, x0=None, verbose=False):
		"""Solves the equation system (S"+N")x = b. This is done in
		a single step if S and N are in the same domain. Otherwise,
		conjugate gradients is used."""
		mode = self.mode.lower()
		if mode[0] == mode[1]:
			# Can solve directly
			return self.M(b)
		else:
			if x0 is None: x0 = b*0
			dof = DOF(Arg(default=self.d))
			def wrap(fun):
				return lambda x: dof.zip(fun(*dof.unzip(x)))

			solver = cg.CG(wrap(self.A), dof.zip(b), x0=dof.zip(x0), M=wrap(self.M))
			for i in range(50):
			#while solver.err > 1e-6:
			#while solver.err_true > 10:
				solver.step()
				if verbose:
					print("%5d %15.7e %15.7e" % (solver.i, solver.err, solver.err_true))
			return dof.unzip(solver.x)[0]
	def wiener(self, verbose=False):
		"""Computes and returns a wiener-filtered version of the
		input map (as set by the constructor or set_data): (S"+N")"N"d."""
		self.x = self.solve(self.Nd, self.x, verbose)
		return self.x
	def sample(self, verbose=False):
		"""Draws a sample s <- P(s|d,S,N), a constrained realization
		of the field based on the input data d and the signal and noise
		covariances."""
		self.b = self.calc_b()
		self.x = self.solve(self.b, self.x, verbose)
		return self.x
	def calc_b(self):
		rand_term = [self.Tmul(c,mat,enmap.rand_gauss(self.d.shape, self.d.wcs)) for c,mat in zip(self.mode, self.hmats)]
		return self.Nd + np.sum(rand_term,0)
	def Tmul(self, char, mat, vec):
		"""Helper function, which handles domain-changing when performing
		sparse matrix-vector multiplication."""
		T = self.trf[char.lower()]
		return T[1](mul(mat,T[0](vec),self.axes))

class NoiseSampler:
	"""Given a set of realizations of a noisy enmap d <- N(s,N),
	draws joint samples of the field's mean and variance. Noise
	is assumed to be independent in each degree of freedom of
	d. d has the form [nsamp,...,ny,nx], and the resulting
	noise model will be [...,ny,nx].
	
	The samples are given by:
		N <- 1/gamma(ndof/2, 1/(chisq/2))
	where ndof is nsamp*(1 for real, 2 for complex).
	"""
	def __init__(self, data):
		chisq = np.sum(np.abs(data-np.mean(data,0)[None])**2,0)
		ndof, scale = data.shape[0]+1, 1
		# Check this more properly. Logically I should multiply
		# by two, since I have twice as many numbers. But that gives
		# the wrong expectation value. Also, I really should subtract
		# one due to removing the mean, but that doesn't work either.
		#if issubclass(data.dtype.type, np.complex):
		#	ndof *= 2; scale = 0.5
		self.chisq, self.ndof, self.dtype, self.scale = chisq, ndof, data.dtype, scale
	def sample(self):
		return 1/np.random.gamma(self.ndof/2.0, 2.0/self.chisq)
