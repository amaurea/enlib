"""This module represents the map-making equation P'N"Px = P'N"d.
At this level of abstraction, we still deal mostly with maps and cuts etc.
directly."""
import numpy as np, bunch, time, h5py, copy
from enlib import pmat, config, nmat, enmap, array_ops, fft, cg, utils
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
		elif precon == "sub":
			self.precon = PrecondSubmap(self.mapeq)
		self.mask   = self.precon.mask
		self.dof    = DOF({"shared":self.mask},{"distributed":self.mapeq.njunk})
		self.b      = self.dof.zip(*self.mapeq.b())
		self.scans, self.area, self.comm = scans, area, comm
		# Store a copy of the next level, which
		# we will use when going up and down in levels.
		self._upsys = None
	def A(self, x):
		#print "linmap A1", np.sum(x**2)
		res = self.dof.zip(*self.mapeq.A(*self.dof.unzip(x)))
		#print "linmap A2", np.sum(res**2)
		return res
	def M(self, x):
		#print "linmap M1", np.sum(x**2)
		res = self.dof.zip(*self.precon.apply(*self.dof.unzip(x)))
		#print "linmap M2", np.sum(res**2)
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
	"""This class implements a simple "binned" preconditioner, which
	disregards detector and time correlations, and solves the system on
	a pixel by pixel basis. It does take into account correlations between
	the different signal components inside each pixel, though."""
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
	"""This preconditioner approximates the A matrix as
	SCS, where S is a position-dependent standard deviation,
	and C is a position-independent correlation pattern.
	It works well for maps with uniform scanning patterns."""
	def __init__(self, mapeq):
		ncomp, h,w = mapeq.area.shape
		binned = PrecondBinned(mapeq)

		S  = array_ops.eigpow(binned.div_map, -0.5, axes=[0,1])
		# Sample 4 points to avoid any pathologies
		N  = 2
		pix = [[h*(2*i+1)/N/2,w*(2*j+1)/N/2] for i in range(N) for j in range(0,N)]
		pix = np.array([[-1,-1],[1,1]])*10+np.array([h/2,w/2])[None,:]
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
		return m, junk/self.div_junk

class PrecondSubmap:
	"""This preconditioner splits the scans into
	subsets with as similar properties as possible.
	For each subset, a good approximation to the pixel
	covariance matrix is constructed, and the submaps are
	then optimally combined using these matrices (in
	practice using conjugate gradients). This is a somewhat
	expensive preconditioner, but will hopefully pay for
	it with much fewer iterations needed.
	"""
	def __init__(self, mapeq):
		# Categorize each scan into groups which can be
		# combined into one large scan with the same
		# scanning pattern as the individual scans.
		scaninfo = []
		for d in mapeq.data:
			scaninfo.append(analyze_scanning_pattern(d))
		allinfo = mapeq.comm.allreduce(scaninfo)
		groups  = group_scans_by_scandirs(allinfo)
		groups  = split_disjoint_scan_groups(groups)
		# For each group, define a single effective
		# scan.
		for group in groups:
			obox_tot = np.array([np.min([g.obox[0] for g in group],0),np.max([g.obox[1] for g in group],0)])
			# We already know that the scanning directions and amplitudes agree.
			# So to construct the total bounds in input coordinates, we just need
			# to take into account the drift vector. First compute the number of drift
			# vectors we are off. out_off = ovec*x, in_off = ivec*x,
			# x = (ovec'ovec)"ovec'out_off
			ivec, iref = group[0].ivecs, group[0].ibox
			ovec, oref = group[0].ovecs, group[0].obox
			x = np.linalg.solve(ovec.T.dot(ovec),ovec.T.dot((obox_tot-oref).T)).T
			ibox_tot = iref + x.dot(ivec)
			# We must now create new effective scan objects. These require:
			#  boresight: fit to ibox_tot, repeating scanning pattern as necessary
			#  offsets:   ncomp detectors, all with same (zero?) offsets
			#  comps:     detectors have only a single comp each
			#  cut:       no cuts
			#  sys, site, mjd0: copy
			# TODO: 1. extract and repeat scanning pattern. Amplitude?
			#          Just use triangle wave based on scan and drift vectors?
			#       2. Identify scan to use as basis.
			#          The problem is that that scan may not be available in this
			#          mpi task. Could add mpi_id, scan_id to scaninfo and then
			#          use mpi_receive or something to get that scan.
			#       3. Distribute groups.


		print [[id(g) for g in group] for group in groups]

		1/0

		# 2. Build sub-map-equations for each of these
		# 3. Build a pixel ordering that closely approximates the
		# order the pixels are hit in the scanning pattern.
		# 4. Build an approximate noise matrix for the TOD implied by
		# that pixel ordering.
		# 5. Build a right-hand-side for each submap based on each
		# sub-map-equation.
	def __apply__(self, map, junk):
		pass
		# 1. Solve the equation sum_sub(A_sub) x = sum_sub b_sub
		# by reading off the pixels from each, unapplying the noise
		# matrix and adding them back to the right position.
		#
		# How to handle polarization? The effective noise correlation
		# length is different for polarization, so making one timestream
		# for each component will be suboptimal unless we build a different
		# noise model for each. On the other hand, projecting them down into
		# a single time-stream won't work because the effective model only
		# has one detector.
		#
		# If we assume that all detectors are hitting the same pixel, then
		# it should be possible to compute effective T and P TOD noise models.
		# n"(f,c1,c2) = N"(f,d1,d2) phase(d1,c1) phase(d2,c2)
		# We here assume that the detector phases are time-independent.
		# N(f,d1,d2) = U(f,d1) delta(d1,d2) + V(f,d1,b) E(f,b) V(f,d2,b)
		# N"(f,d1,d2)= iU(f,d1) delta(d1,d2) - Q(f,d1,b)Q(f,d2,b)
		# The interaction between phase and Q will ensure that the polarized
		# noise ends up lower.

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

def analyze_scanning_pattern(d):
	"""Computes bounding boxes for d.scan in both input and output
	coordinates. Also computes the scan and drift vectors. Returns
	bunch(ibox, obox, ivecs, ovecs). The ivecs and ovecs are in units
	per sample."""
	box   = d.scan.box
	nidim = box.shape[1]
	# First determine the scanning direction. This is the dimension
	# that is responsible for the fastest change in output coordinates,
	# but a simpler way of finding it is to find the direction with the
	# most sign changes in the derivative.
	nsamp   = d.scan.boresight.shape[0]
	periods = utils.find_period(d.scan.boresight.T)
	dscan   = np.argmin(periods)
	# The remaining dimensions are assumed to change uniformly.
	# ivecs[0] is the scanning vector, ivecs[1] is the drift vector
	ivecs = np.array([box[1]-box[0],box[1]-box[0]])
	ivecs[1,dscan] = 0
	ivecs[0,np.arange(nidim)!=dscan] = 0
	# Normalize ivecs so that they show the typical step per pixel
	# The 2 is because we cross the full distance twice in one period
	# (unless this is a wrapping scan, but we assume that isn't the case)
	ivecs[0] /= periods[dscan]/2
	ivecs[1] /= nsamp

	# Translate these input vectors to the output coordinate system
	mid   = np.mean(box,0)
	bore  = np.array([mid+v for v in [0] + list(ivecs)])
	pix, _= d.pmap.translate(bore)
	ovecs = np.array([pix[0,i+1]-pix[0,0] for i in range(2)])

	# Compute an approximate bounding box in output coordinates
	obox = d.pmap.translate(box)[0][0]

	return bunch.Bunch(ibox=box, ivecs=ivecs, obox=obox, ovecs=ovecs, scandim=dscan)

def group_scans_by_scandirs(info, vectol=0.1, postol=0.1):
	# Group them into overlapping groups with
	# consistent scanning directions and consistent
	# position in the scan direction.
	unclassified = range(len(info))
	groups = []
	while len(unclassified) > 0:
		me = info[unclassified.pop()]
		veclens = np.sum(me.ovecs**2,1)**0.5
		poslens = np.abs(me.obox[1]-me.obox[0])
		accepted = []
		for oi in unclassified:
			other = info[oi]
			vecdiff = np.sum((me.ovecs-other.ovecs)**2,1)**0.5
			if np.any(vecdiff > veclens*vectol): continue
			# Ok, the scanning directions match.
			# Check that the positions also match. The range in position
			# in the scanning direction must be the same.
			posdiff = np.linalg.solve(me.ovecs.T.dot(me.ovecs),me.ovecs.T.dot((me.obox-other.obox).T))
			posdiff = np.sum(posdiff**2,1)**0.5
			if posdiff[0] > poslens[0]*postol: continue
			accepted.append(oi)
		mygroup = [me]
		for v in accepted:
			mygroup.append(info[v])
			unclassified.remove(v)
		groups.append(mygroup)
	return groups

def split_disjoint_scan_groups(groups):
	# 3. Split disjoint groups into contiguous subgroups
	subgroups = []
	for group in groups:
		print "A", len(group)
		# Compute from-to for each in the drift direction
		driftvec = group[0].ovecs[1]
		driftvec /= np.sum(driftvec**2)**0.5
		pos = np.array([np.sum(member.obox*driftvec,1) for member in group])
		# Sort group by the starting position
		inds = np.argsort(pos[:,0])
		group, pos = [group[i] for i in inds], pos[inds]
		print "B", pos
		# Then split into non-touching regions
		sub = []
		end = pos[0,1]
		for p, member in zip(pos,group):
			if p[0] > end:
				if len(sub) > 0: subgroups.append(sub)
				sub = []
			end = max(end,p[1])
			sub.append(member)
		if len(sub) > 0: subgroups.append(sub)
	return subgroups
