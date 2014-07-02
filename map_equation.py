"""This module represents the map-making equation P'N"Px = P'N"d.
At this level of abstraction, we still deal mostly with maps and cuts etc.
directly."""
import numpy as np, bunch, time, h5py
from enlib import pmat, config, nmat, enmap, array_ops, fft, cg
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
	def __init__(self, scans, area, comm=MPI.COMM_WORLD, precon="bin"):
		self.mapeq  = MapEquation(scans, area, comm=comm)
		if precon == "bin":
			self.precon = PrecondBinned(self.mapeq)
		elif precon == "cyc":
			self.precon = PrecondCirculant(self.mapeq)
		self.mask   = self.precon.mask
		self.dof    = DOF({"shared":self.mask},{"distributed":self.mapeq.njunk})
		self.b      = self.dof.zip(*self.mapeq.b())
		self.scans, self.area, self.comm = scans, area, comm
		# Store a copy of the next level, which
		# we will use when going up and down in levels.
		self._upsys = None
	def A(self, x):
		print "linmap A1", np.sum(x**2)
		res = self.dof.zip(*self.mapeq.A(*self.dof.unzip(x)))
		print "linmap A2", np.sum(res**2)
		return res
	def M(self, x):
		print "linmap M1", np.sum(x**2)
		res = self.dof.zip(*self.precon.apply(*self.dof.unzip(x)))
		print "linmap M2", np.sum(res**2)
		return res
	def dot(self, x, y): return self.dof.dot(x,y)
	@property
	def upsys(self):
		if self._upsys is None:
			# Produce a downgraded equation set, where spatial
			# and temporal resolution is halved.
			scans = [scan[:,::2] for scan in self.scans]
			area  = self.area[:,::2,::2]
			self._upsys = LinearSystemMap(scans, area, self.comm)
		return self._upsys
	def up(self, x):
		# Downgrade the vector x.
		hmap, hjunk = self.dof.unzip(x)
		lmap, ljunk = self.upsys.dof.unzip(np.empty(self.upsys.dof.n,dtype=x.dtype))
		pmat.PmatMapRebin().forward(hmap,lmap)
		for hdata, ldata in zip(self.mapeq.data, self.upsys.mapeq.data):
			rebin = pmat.PmatCutRebin(hdata.pcut, ldata.pcut)
			rebin.forward(hjunk[hdata.cutrange[0]:hdata.cutrange[1]], ljunk[ldata.cutrange[0]:ldata.cutrange[1]])
		return self.upsys.dof.zip(lmap,ljunk)
	def down(self, x):
		# Upgrade the vector x to the resolution of the current level
		hmap, hjunk = self.dof.unzip(np.empty(self.dof.n,dtype=x.dtype))
		lmap, ljunk = self.upsys.dof.unzip(x)
		pmat.PmatMapRebin().backward(hmap,lmap)
		for hdata, ldata in zip(self.mapeq.data, self.upsys.mapeq.data):
			rebin = pmat.PmatCutRebin(hdata.pcut, ldata.pcut)
			rebin.backward(hjunk[hdata.cutrange[0]:hdata.cutrange[1]], ljunk[ldata.cutrange[0]:ldata.cutrange[1]])
		return self.dof.zip(hmap,hjunk)

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
		map, junk = map.copy(), junk.copy()
		omap, ojunk = map*0, junk*0
		for d in self.data:
			tod = np.empty([d.scan.ndet,d.scan.nsamp],dtype=self.dtype)
			d.pmap.forward(tod,map)
			d.pcut.forward(tod,junk[d.cutrange[0]:d.cutrange[1]])
			if white:
				d.nmat.white(tod)
			else:
				d.nmat.apply(tod)
			d.pcut.backward(tod,ojunk[d.cutrange[0]:d.cutrange[1]])
			d.pmap.backward(tod,omap)
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

class PrecondCirculant:
	def __init__(self, mapeq):
		ncomp, h,w = mapeq.area.shape
		binned = PrecondBinned(mapeq)

		S  = array_ops.eigpow(binned.div_map, -0.5, axes=[0,1])
		N  = 2
		pix = [[h*(2*i+1)/N/2,w*(2*j+1)/N/2] for i in range(N) for j in range(i,N)]
		Arow = measure_corr_cyclic(mapeq, S, pix)
		iC = np.conj(fft.fft(Arow, axes=[-2,-1]))

		self.S, self.iC = S, iC
		self.div_junk = binned.div_junk
		self.mask = binned.mask
	def apply(self, map, junk):
		# We will apply the operation m \approx S C S map
		# The fft normalization is baked into iC.
		m  = array_ops.matmul(self.S, map, axes=[0,1])
		mf = fft.fft(m, axes=[-2,-1])
		mf = array_ops.solve_masked(self.iC, mf, axes=[0,1])
		m  = fft.ifft(mf, axes=[-2,-1]).astype(map.dtype)
		m/= np.prod(m.shape[-2:])
		m  = array_ops.matmul(self.S, m,   axes=[0,1])
		#print np.min(m), np.max(m)
		#with h5py.File("test.hdf","w") as hfile:
		#	hfile["mfR"] = mf.real
		#	hfile["mfI"] = mf.imag
		#	hfile["m"]   = m
		return m, junk/self.div_junk


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

def measure_Arow(mapeq, pix):
	ncomp,h,w = mapeq.area.shape
	Arow = enmap.zeros([ncomp,ncomp,h,w],mapeq.area.wcs, dtype=mapeq.area.dtype)
	junk = np.zeros(mapeq.njunk, dtype=mapeq.area.dtype)
	for ci in range(ncomp):
		Arow[ci,ci,pix[0],pix[1]] = 1
		junk[...]         = 0
		Arow[ci], _ = mapeq.A(Arow[ci], junk)
	return Arow

def cov2corr(iC, S, ref, beam):
	print "A", np.max(iC), np.min(iC)
	Sref = S.copy(); S[...] = S[:,:,ref[0],ref[1]][:,:,None,None]
	iC = array_ops.matmul(iC,    S, axes=[0,1])
	iC = array_ops.matmul(Sref, iC, axes=[0,1])
	print "B", np.max(iC), np.min(iC)
	# Shift the reference pixel to 0,0:
	iC = np.roll(iC, -ref[0], 2)
	iC = np.roll(iC, -ref[1], 3)
	# Regularize by damping long-distance correlations
	if beam > 0: apply_gaussian(iC, beam)
	# And store in fourier domain
	res = np.conj(fft.fft(iC, axes=[-2,-1]))
	## Overnormalize in order to avoid later normalization
	#res /= np.prod(iC.shape[-2:])**2
	return res

def apply_gaussian(fa, sigma):
	flat  = fa.reshape(-1,fa.shape[-2],fa.shape[-1])
	gauss = [np.exp(-0.5*(np.arange(n)/sigma)**2) for n in flat.shape[-2:]]
	gauss = [g + g[::-1] for g in gauss]
	flat *= gauss[0][None,:,None]
	flat *= gauss[1][None,None,:]

def measure_corr_cyclic(mapeq, S, pixels):
	# Measure the typical correlation pattern by using multiple
	# pixels at the same time.
	ncomp,h,w = mapeq.area.shape
	d = enmap.zeros([ncomp,ncomp,h,w],mapeq.area.wcs, dtype=mapeq.area.dtype)
	junk = np.zeros(mapeq.njunk, dtype=mapeq.area.dtype)
	for p in pixels:
		Arow = d*0
		for ci in range(ncomp):
			#Arow[ci,:,p[0],p[1]] = S[ci,:,p[0],p[1]]
			Arow[ci,ci,p[0],p[1]] = 1
			junk[...] = 0
			Arow[ci,:],_ = mapeq.A(Arow[ci], junk)
		Sref = S.copy(); S[...] = S[:,:,p[0],p[1]][:,:,None,None]
		Arow = array_ops.matmul(Arow, S,    axes=[0,1])
		Arow = array_ops.matmul(Sref, Arow, axes=[0,1])
		Arow = np.roll(Arow, -p[0], 2)
		Arow = np.roll(Arow, -p[1], 3)
		d += Arow
	d /= len(pixels)
	return d

#	for ci in range(ncomp):
#		for p in pixels:
#			d[ci,:,p[0],p[1]] = S[ci,:,p[0],p[1]]
#		junk[...]         = 0
#		d[ci], _ = mapeq.A(d[ci], junk)
#	d = array_ops.matmul(S, d, axes=[0,1])
#	# d now holds a linear combination of various rows from the matrix (SAS).
#	# We wish to disentangle these to get an estimate "a" for the typical
#	# correlation around each pixel, assuming this is the same for all pixels.
#	# To do this we solve the system Pa = d, where P represents the operation
#	# that takes us from the real correlation structure centered on pixel (0,0)
#	# to the superimposed pattern we have now.
#	def P(a, dir=1):
#		d = a*0
#		for p in pixels:
#			tmp = np.roll(a,   -p[0]*dir, 2)
#			tmp = np.roll(tmp, -p[1]*dir, 3)
#			d += tmp
#		return d
#	def PT(d): return P(d, dir=-1)
#	dof = DOF(np.isfinite(d))
#	def A(x): return dof.zip(PT(P(*dof.unzip(x))))
#	b = dof.zip(d)
#	solver = cg.CG(A, b)
#	while True:
#		solver.step()
#		print "mcc %4d %15.7e" % (solver.i, solver.err)
#		x, = dof.unzip(solver.x)
#		with h5py.File("mcc%03d.hdf" % solver.i,"w") as hfile:
#			hfile["data"] = x
#	return dof.unzip(solver.x)
