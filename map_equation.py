"""This module represents the map-making equation P'N"Px = P'N"d.
At this level of abstraction, we still deal mostly with maps and cuts etc.
directly."""
import numpy as np, bunch, time
from enlib import pmat, config, nmat, enmap, array_ops
from enlib.degrees_of_freedom import DOF
from mpi4py import MPI

class LinearSystem:
	def A(self): raise NotImplementedError
	def M(self): raise NotImplementedError
	def b(self): raise NotImplementedError
	def dot(self, x, y): raise NotImplementedError
	def expand(self, x): raise NotImplementedError
	def flatten(self, params): raise NotImplementedError
	def level(self): return 0
	def up(self, x=None): raise NotImplementedError
	def down(self): raise NotImplementedError

# Abstract interface to the Map-making system.
class LinearSystemMap(LinearSystem):
	def __init__(self, scans, area, comm=MPI.COMM_WORLD):
		self.mapeq  = MapEquation(scans, area, comm=comm)
		self.precon = PrecondBinned(self.mapeq)
		self.mask   = self.precon.mask
		self.dof    = DOF({"shared":self.mask},{"distributed":self.mapeq.njunk})
		self.b      = self.dof.zip(*self.mapeq.b())
	def A(self, x):
		return self.dof.zip(*self.mapeq.A(*self.dof.unzip(x)))
	def M(self, x):
		return self.dof.zip(*self.precon.apply(*self.dof.unzip(x)))

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

class MapEquation:
	def __init__(self, scans, area, comm=MPI.COMM_WORLD, pmat_order=None, cut_type=None, eqsys=None):
		data = []
		njunk = 0
		for scan in scans:
			d = bunch.Bunch()
			d.scan = scan
			d.pmap = pmat.PmatMap(scan, area, order=pmat_order, sys=eqsys)
			d.pcut = pmat.PmatCut(scan, cut_type)
			d.cutrange = [njunk,njunk+d.pcut.njunk]
			njunk = d.cutrange[1]
			# This should be configurable
			d.nmat = nmat.NmatDetvecs(scan.noise)
			data.append(d)
		self.area = area.copy()
		self.njunk = njunk
		self.dtype = area.dtype
		self.comm  = comm
		self.data  = data
	def b(self):
		rhs_map  = enmap.zeros(self.area.shape, self.area.wcs, dtype=self.dtype)
		rhs_junk = np.zeros(self.njunk, dtype=self.dtype)
		for d in self.data:
			tod = d.scan.get_samples()
			tod-= np.mean(tod,1)[:,None]
			tod = tod.astype(self.dtype)
			d.nmat.apply(tod)
			d.pmap.backward(tod,rhs_map)
			d.pcut.backward(tod,rhs_junk[d.cutrange[0]:d.cutrange[1]])
		return reduce(rhs_map, self.comm), rhs_junk
	def A(self, map, junk, white=False):
		t1 = time.time()
		map, junk = map.copy(), junk.copy()
		omap, ojunk = map*0, junk*0
		t2 = time.time()
		for d in self.data:
			tod = np.empty([d.scan.ndet,d.scan.nsamp],dtype=self.dtype)
			t3 = time.time()
			d.pmap.forward(tod,map)
			t4 = time.time()
			d.pcut.forward(tod,junk[d.cutrange[0]:d.cutrange[1]])
			t5 = time.time()
			if white:
				d.nmat.white(tod)
			else:
				d.nmat.apply(tod)
			t6 = time.time()
			d.pcut.backward(tod,junk[d.cutrange[0]:d.cutrange[1]])
			t7 = time.time()
			d.pmap.backward(tod,omap)
			t8 = time.time()
		print "i: %4.1f t: %4.1f mf: %4.1f jf: %4.1f n: %4.1f jb: %4.1f mb: %4.1f" % (t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7)
		return reduce(omap, self.comm), ojunk
	def white(self, map, junk):
		return self.A(map, junk, white=True)

class PrecondBinned:
	def __init__(self, mapeq):
		ncomp     = mapeq.area.shape[0]
		# Compute the per pixel approximate inverse covmat
		div_map   = enmap.zeros((ncomp,ncomp)+mapeq.area.shape[1:],mapeq.area.wcs, mapeq.area.dtype)
		div_junk  = np.zeros(mapeq.njunk, dtype=mapeq.area.dtype)
		for ci in range(ncomp):
			div_map[ci,ci] = 1
			div_junk[...]  = 1
			div_map[ci], div_junk = mapeq.white(div_map[ci], div_junk)
		self.div_map, self.div_junk = reduce(div_map, mapeq.comm), div_junk
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

def reduce(a, comm=MPI.COMM_WORLD):
	res = a.copy()
	comm.Allreduce(a, res)
	return res
