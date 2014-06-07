"""This module represents an abstract interface for linear systems, which
encapsulates all the information needed to define and solve it, presenting
a uniform interface that various solution methods can use."""
import numpy as np
from enlib import pmat, config
from mpi4py import MPI

#class LinearSystem:
#	def A(self): raise NotImplementedError
#	def M(self): raise NotImplementedError
#	def b(self): raise NotImplementedError
#	@parameter
#	def dot(self, x, y): raise NotImplementedError
#	def expand(self, x): raise NotImplementedError
#	def flatten(self, params): raise NotImplementedError
#	def level(self): return 0
#	def up(self, x=None): raise NotImplementedError
#	def down(self): raise NotImplementedError

# FIXME: How should I get the noise matrix? As an argument?
# Should it already be measured, or should it be measured internally?
# If already measured, one would have to pass an array of nmats
# of the same length as scans. That's probably best - we may
# want to move to storing nmats in files at some point.
# Perhaps the constructor should take a single array with
# entries of [.scan, .pmap, .pcut, .nmat]. That would
# free us up to make this more general, and would let us put
# the part that creates the argument array in a part of the
# code that is allowed to use enact stuff.
#
# Yes, let's plan for noise matrices being read from disk
# and stored as an element in Scan. Then scan contains all
# the information needed to initialize a MapSystem.
#
# There is a circular dependency the way I'm doing this here
# M(x) depends on A(x), but the A(x) interpretation of the plain
# array x depends on the mask derived from M(x).
#
# I think the solution is to seperate the plain array (x) stuff
# (which needs masking, flattening and expanding) from the
# underlying A(map,junk), M(map), M(junk), etc.
# This lower layer will be tied to maps etc. while the upper level
# is a general abstraction. Built from these.

# FIXME: This file should be named map_equation.py, and should represent
# everything we need about the map-maker equation.

def reduce(a, comm=MPI.COMM_WORLD):
	res = a.copy()
	comm.Allreduce(a, res)
	return res

class MapSystem:
	def __init__(self, scans, area, comm=MPI.COMM_WORLD, pmat_order=None, cut_type=None, eqsys=None):
		data = []
		njunk = 0
		for scan in scans:
			d = bunch.Bunch()
			d.scan = scan
			d.pmap = PmatMap(scan, area, order=pmat_order, sys=eqsys)
			d.pcut = PmatCut(scan, cut_type)
			d.cutrange = [njunk,njunk+d.pcut.njunk]
			njunk = d.cutrange[1]
			# For this to work, NmatDetvecs must be available in enlib
			d.nmat = NmatDetvecs(scan)
			data.append(d)
		self.area = area.copy()
		self.njunk = njunk
		self.dtype = area.dtype
		self.comm  = comm
	def b(self):
		rhs_map  = enmap.zeros(self.area.shape, self.area.wcs, dtype=self.dtype)
		rhs_junk = np.zeros(self.njunk, dtype=self.dtype)
		for d in data:
			tod = d.scan.get_samples()
			d.pmap.backward(tod,rhs_map)
			d.pcut.backward(tod,rhs_junk[d.cutrange[0]:d.cutrange[1]])
		return reduce(rhs_map, self.comm), rhs_junk
	def A(self, map, junk, white=False):
		map, junk = map.copy(), junk.copy()
		for d in data:
			tod = np.empty([d.scan.ndet,d.scan.nsamp],dtype=self.dtype)
			d.pmap.forward(tod,map)
			d.pcut.forward(tod,junk)
			if white:
				d.nmat.white(tod)
			else:
				d.nmat.apply(tod)
			d.pcut.backward(tod,junk)
			d.pcut.backward(tod,map)
		return reduce(map, self.comm), junk
	def white(self, map, junk):
		return self.A(map, junk, white=True)

class PrecondBinned:
	def __init__(self, system):
		ncomp     = system.area.shape[0]
		# Compute the per pixel approximate inverse covmat
		div_map   = enmap.zeros((ncomp,ncomp)+system.area.shape[1:],system.area.wcs, system.area.dtype)
		div_junk  = np.zeros(system.njunk, dtype=system.area.dtype)
		for ci in range(ncomp):
			div_map[ci,ci] = 1
			div_junk[...]  = 1
			div_map[ci], div_junk = system.white(div_map[ci], div_junk)
		self.div_map, self.div_junk = reduce(div_map, system.comm), div_junk
		# Compute the pixel component masks, and use it to mask out the
		# corresonding parts of the map preconditioner
		self.mask = makemask(self.div_map)
		self.div_map *= self.mask[None,:]*self.mask[:,None]
	def apply(self, map, junk):
		return array_ops.solve_masked(self.div_map, map, [0,1]), junk/self.div_junk

config.default("precond_condition_lim", 10., "Maximum allowed condition number in per-pixel polarization matrices.")
def makemask(div):
	condition = array_ops.condition_number_multi(div, [0,1])
	tmask = div[0,0] > 0
	lim   = config.get("precond_condition_lim")
	pmask = (condition >= 1)*(condition < lim)
	masks = enmap.zeros(div.shape[1:], div.wcs, dtype=bool)
	masks[0]  = tmask
	masks[1:] = pmask[None]
	return masks


		# And the preconditioner
		# FIXME: Here we encounter the problem that some preconditioners
		# require a working A-operator to work. Since I need the preconditioner
		# to define a MapSystem, I need to be able to initialize an A-operator
		# independently of it to get the prconditioner first.



