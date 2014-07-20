import numpy as np
from enlib import enmap, array_ops, cg
from enlib.degrees_of_freedom import DOF

def mul(mat, vec, axes):
	return array_ops.matmul(mat.astype(vec.dtype),vec,axes=axes)
def pow(mat, exp, axes): return array_ops.eigpow(mat,exp,axes=axes)

class FieldSampler:
	"""Draws samples from P(s|d,S,N), under the model d <- N(0,S+N).
	iS and iN must be (...,ncomp,ncomp,ny,nx), while d is
	(...,ncomp,ny,nx). "i" indicates the inverse here. The axes
	argument can be used to change which axes are regarded as the
	component directions. The "desc" argument describes the basis
	iS and iN are expesse in. The first letter corresponds to iS,
	the second to iN. "h" means harmonic space, and "p" means
	pixel space. Capitalized letters indicates which space to use
	for the internal preconditioner.

	The actual sampling is done using the .sample() method.
	To get predictable values, use np.random.seed(). The class
	can also be used to wiener-filter, using the .wiener() method."""
	def __init__(self, iS, iN, data=None, desc="Hp", axes=[-4,-3]):
		# Desc describes the signal and noise domains. It is
		# a two-character string, where the first character describes
		# S and the second N. The characters are "p" for pixel-space
		# and "h" for harmonic space. The maps are always in pixel-space.
		self.desc=desc
		self.imats = [iS, iN]
		self.hmats = [pow(imat, 0.5, axes=axes) for imat in self.imats]
		self.axes  = axes
		self.trf = {
				"p":(lambda x:x, lambda x:x),
				"h":(enmap.map2harm, enmap.harm2map)
			}
		# Preconditioner
		self.pind = [c.isupper() for c in desc+"F"].index(True)%2
		if desc.lower()[0] == desc.lower()[1]:
			self.prec = pow(np.sum(self.imats,0),-1,axes=axes)
		else:
			white= np.mean(np.mean(self.imats[1-self.pind],-1),-1)
			self.prec = pow(self.imats[self.pind]+white.T[None,None].T,-1,axes=axes)
		self.x, self.b = None,None
		self.set_data(data)
	def M(self, vec):
		return self.Tmul(self.desc[self.pind], self.prec, vec)
	def A(self, vec):
		desc = self.desc.lower()
		if desc[0] == desc[1]:
			return self.Tmul(desc[0], np.sum(self.imats,0), vec)
		else:
			Av1 = self.Tmul(desc[0], self.imats[0], vec)
			Av2 = self.Tmul(desc[1], self.imats[1], vec)
			return Av1 + Av2
	def set_data(self, data):
		self.d = data
		self.Nd = self.Tmul(self.desc[1], self.imats[1], data) if data is not None else None
	def solve(self, b, x0=None, verbose=False):
		desc = self.desc.lower()
		if desc[0] == desc[1]:
			# Can solve directly
			return self.M(b)
		else:
			if x0 is None: x0 = b*0
			dof = DOF(self.d)
			def wrap(fun):
				return lambda x: dof.zip(fun(*dof.unzip(x)))
			solver = cg.CG(wrap(self.A), dof.zip(b), x0=dof.zip(x0), M=wrap(self.M))
			while solver.err > 1e-6:
				solver.step()
				if verbose:
					print "%5d %15.7e %15.7e" % (solver.i, solver.err, solver.err_true)
			return dof.unzip(solver.x)[0]
	def wiener(self, verbose=False):
		self.x = self.solve(self.Nd, self.x, verbose)
		return self.x
	def sample(self, verbose=False):
		self.b = self.calc_b()
		self.x = self.solve(self.b, self.x, verbose)
		return self.x
	def calc_b(self):
		rand_term = [self.Tmul(c,mat,enmap.rand_gauss(self.d.shape, self.d.wcs)) for c,mat in zip(self.desc, self.hmats)]
		return self.Nd + np.sum(rand_term,0)
	def Tmul(self, char, mat, vec):
		T = self.trf[char.lower()]
		return T[1](mul(mat,T[0](vec),self.axes))
