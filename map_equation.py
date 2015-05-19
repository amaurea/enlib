"""This module represents the map-making equation P'N"Px = P'N"d.
At this level of abstraction, we still deal mostly with maps and cuts etc.
directly."""
import numpy as np, bunch, time, h5py, copy, logging, sys
from enlib import pmat, config, nmat, enmap, array_ops, fft, cg, utils, rangelist, scansim
from enlib import bench, todfilter, dmap, zipper
from scipy import ndimage
from mpi4py import MPI

L = logging.getLogger(__name__)

# Map parallelization
# ===================
# With the large patches, a significant fraction of the time
# is being spent on even the simple preconditioner, and the maps
# are also starting to take up a large amount of space. This
# prevents us from scaling to larger clusters and larger patches.
# We need to spread the pixels out, like we do with the samples.
#
# 1. Divide the full pixel space into blocks. Each block has a
#    single mpi task as its owner, but each block will contain
#    samples from multiple mpi tasks.
#    a) Each task needs to know who owns the blocks it hits
#    b) Each task needs to know who hits each of its blocks
# 2. Because there is no 1-1 mapping from tods to pixels, each
#    mpi tasks needs a local pixel workspace that is larger than
#    the blocks it owns, and has room for all its TODs.
#    Parts that fall outside its ownership must be communicated
#    to the owner.
# 3. Our local workspace should be as small as possible to avoid
#    wasting memory, and to avoid random access slowness. Therefore
#    the TODs each mpi task owns should be as close together as possible.
# 4. We also want each mpi task to take the same time for each step,
#    so they don't have to wait for each other.
#
# Given a set of bounding boxes, our task is basically to determine
# a partitioning of those bounding boxes such that the maximum total
# box is minimized. But we also want the total time difference to
# be minimized too. This requires us to try for a compromise. Define
# score(areas,times) = A*max(areas)+B*max(times). We wish to minimize
# this score.
#
# How will a greedy algorithm perform here?  Start with N bins.
# Sort each box by the score it would have in isolation, with the
# highest scores first. For each box, sompute what score putting
# it in each bin would reulst in. Then assign it to the bin that
# minimizes that score. Repeat until done.
#
# This algorithm sounds pretty straighforward, and will probably give
# pretty good results, though not optimal ones.
#
# After determininig which MPI task owns which TODs, determine the
# task that owns the most tods in each block. Assign ownership to
# that task.
#
# The overall cycle is then:
#
# A matrix
#  1. transfer from blocks to workspaces (mpi communication)
#  2. project from workspaces to tod
#  3. apply noise matrix
#  4. project from tod to workspaces
#  5. coadd workspaces into blocks
#
# dot product
#  1. compute dot of my blocks
#  2. mpi reduce across tasks
#
# Binned preconditioner:
#  1. apply to the blocks I own
#
# Cyclic preconditioner needs some thinking. We probably don't
# want to apply it globally, since that would require enormous
# FFTs, and data ownership for FFTs doesn't really mesh with
# the blocks I want to use. Apply a limited-range version of a
# reduced pixel space, and stitch? That failed before, but perhaps
# I can get it to work.

class LinearSystem:
	def A(self): raise NotImplementedError
	def M(self): raise NotImplementedError
	def b(self): raise NotImplementedError
	def dot(self, x, y): raise NotImplementedError
	def level(self): return 0
	def up(self, x=None): raise NotImplementedError
	def down(self): raise NotImplementedError

# In general we want to be able to
#  1. Solve for a list of signals in paralell
#  2. Subtract from the TOD when building RHS, or replace it with something else.
#  3. Don't make any unnecessary overhead. Make no
#     more passes through the data than necessary.
# I suggest an interface something like this:
#   eq = LinearSystemScans(scans, signals=[sky, cuts, azbin], data=[raw,srcsub])
# sky, cuts, azbin etc. are objects containing enough informatino to solve for
# these components. They should each have members
#  .name # Name for identifying this component
#  .dof  # DOF object representing its degrees of freedom
#  .hits # hitcount map per degree of freedom. Can be expanded using dof
#  .div  # white noise estimate
#  .b    # right-hand size. Empty before being built
#  .M(x) # preconditioner, approximate solution of system
#  .P (s, x, tod) # project from signal to tod for scan s
#  .PT(s, tod, x) # project from tod to signal for scan s
#  .init(scans) # Perform any initialization needed. Pass in all scans
#
# These objects should construct the preconditioner internally. But how to do
# that? The binned preconditioner needs P'NdiagP, which is relatively cheap,
# while the cyclic one needs P'NP. For all the other stuff, these objects
# have not needed to care about N at all, that was the job of the LinearSystemScan
# itself. But for the preconditioner N matters.
#
# Now, technically N *is* available as part of scan. So perhaps this isn't
# a problem. The job of LinearSystemScan is to do all the fancy joint, subtracted
# solving in a CG-friendly manner. That doesn't mean that internal stuff can't
# do extra full loops the scans when building the preconditioner. But when should
# this happen? Easiest thing: .init(scans), which sets up the preconditinoer as needed.
#
# Building the right-hand side:
#
# for scan in scans:
# 	tod = zeros
# 	for dsource in data:
# 		dsource(scan, tod)
# 	nmat.apply(tod)
# 	for signal in signals:
# 		signal.PT(scan, tod, signal.b)
# for signal in signals:
# 	signal.b = signal.dof.reduce(signal.b)
# b = tot_dof.zip(*sum([signal.dof.unzip(signal.b) for signal in signals]))
#
# Memory overhead. The scheme above stores the map redundantly, both as part of
# global array and as part of a local one for one of the signals. This wastes a bit
# of memory, up to a few hundred megabytes for a boss map.
#
# The sources are objects that need to provide __call__(scan, tod) and modifies
# tod. For example:
# def raw(scan, tod): tod[...] = scan.get_samples()
#
# PROBLEMS WITH THIS SCHEME:
#  * Cuts should be taken into account in preconditioner and hitcount map for
#    each signal, but should only correspond to a single signal itself
#  * Preconditioners need to access each signal as a full equation system, so
#    must define full A for the signals and pass them to precon.

# Abstract interface to the Map-making system.
class LinearSystemMap(LinearSystem):
	def __init__(self, scans, area, comm=MPI.COMM_WORLD, precon="bin", imap=None, isrc=None, azmap=None, azfilter=None, subinds=None):
		L.info("Building preconditioner")
		self.mapeq  = MapEquation(scans, area, comm=comm, imap=imap, isrc=isrc, azmap=azmap, azfilter=azfilter, subinds=subinds)
		if precon == "bin":
			self.precon = PrecondBinned(self.mapeq)
		elif precon == "cyc":
			self.precon = PrecondCirculant(self.mapeq)
		elif precon == "sub":
			# This introduces a circular dependency, as PrecondSubmap constructs
			# a LinearSystemMap. This can be broken by sending in a class rather
			# than a string for the precon argument. The current way works. But
			# prevents factorizing out the preconditioners.
			self.precon = PrecondSubmap(self.mapeq)
		elif precon == "symcheck":
			test_symmetry(self.mapeq, 50)
			sys.exit(0)
		elif precon == "symtot":
			prec = PrecondCirculant(self.mapeq)
			test_symmetry(self.mapeq, 100, prec=prec)
			sys.exit(0)
		elif precon == "dumpcov":
			test_symmetry(self.mapeq, 0, verbose=False, shuf=False)
			sys.exit(0)
		self.mask   = self.precon.mask
		zippers = [
				dmap.DmapZipper(area,self.mask),
				zipper.ArrayZipper(np.zeros(self.mapeq.njunk,dtype=area.dtype),shared=False,comm=comm)
				]
		if azmap:
			zippers.append(zipper.ArrayZipper(np.zeros(self.mapeq.azshape,dtype=area.dtype),shared=azmap.shared,comm=comm))
		self.dof = zipper.MultiZipper(zippers, comm=comm)
		L.info("Building right-hand side")
		self.b      = self.dof.zip(*self.mapeq.b())
		self.scans, self.area, self.comm = scans, area, comm
		self.isrc   = isrc
		# Store a copy of the next level, which
		# we will use when going up and down in levels.
		self._upsys = None
	def A(self, x):
		res = self.dof.zip(*self.mapeq.A(*self.dof.unzip(x)))
		return res
	def M(self, x):
		res = self.dof.zip(*self.precon.apply(*self.dof.unzip(x)))
		return res
	def dot(self, x, y):
		res = self.dof.dot(x,y)
		return res
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
	def write(self, prefix="", ext="fits"):
		rhs = self.dof.unzip(self.b)[0]
		dmap.write_map(prefix + "rhs." + ext, rhs)
		self.precon.write(prefix, ext=ext)
		if self.isrc:
			dmap.write_map(prefix + "srcs." + ext, self.isrc.map)

class MapEquation:
	def __init__(self, scans, area, comm=MPI.COMM_WORLD, pmat_order=None, cut_type=None, eqsys=None, imap=None, isrc=None, azmap=None, azfilter=None, subinds=None):
		# Adding ad-hoc simultaneous solving for an azimuth signal. This should really be done
		# in a more ordely fashion, which mapmaking.py will do. But I'm adding it here first
		# as a quick test. azmap should have the following members if present:
		# .npix (int), .shared (bool), .mode ("azimuth", "phase" or None for default)
		data = []
		njunk = 0
		for si, scan in enumerate(scans):
			d = bunch.Bunch()
			d.scan = scan
			# Subinds indicates which local workspace to use for this scan
			d.sub = 0 if subinds is None else subinds[si]
			try:
				d.pmap = pmat.PmatMap(scan, area.work[d.sub], order=pmat_order, sys=eqsys)
			except OverflowError:
				L.debug("Failed to set up pointing interpolation for scan #%d. Skipping" % si)
				continue
			d.pcut = pmat.PmatCut(scan, cut_type)
			d.cutrange = [njunk,njunk+d.pcut.njunk]
			njunk = d.cutrange[1]
			d.nmat = scan.noise
			# Make maps from data projected from input map instead of real data
			if imap: d.pmat_imap = pmat.PmatMap(scan, imap.work[d.sub], order=pmat_order, sys=imap.sys)
			if isrc: d.pmat_isrc = pmat.PmatPtsrc(scan, isrc.model.params.astype(area.dtype), sys=isrc.sys, tmul=isrc.tmul, pmul=isrc.pmul)
			if azmap: d.pmat_azmap = pmat.PmatScan(scan, area.shape[0], azmap.npix, mode=azmap.mode)
			data.append(d)
		self.area = area.copy()
		self.njunk = njunk
		self.dtype = area.dtype
		self.comm  = comm
		self.data  = data
		self.imap  = imap
		self.isrc  = isrc
		self.azfilter = azfilter
		self.azmap = azmap
		if self.azmap:
			npre = 1 if azmap.shared else len(scans)
			self.azshape = (npre, area.shape[0], azmap.npix)
	def b(self):
		rhs_map  = self.area.copy()
		rhs_junk = np.zeros(self.njunk, dtype=self.dtype)
		if self.azmap:
			rhs_azmap= np.zeros(self.azshape, dtype=self.dtype)
		for di, d in enumerate(self.data):
			azdi = 0 if not self.azmap or len(rhs_azmap) <= 1 else di
			with bench.mark("meq_b_get"):
				# Only read data if necessary, as it's a pretty heavy operation
				if (self.imap is None or self.imap.tmul != 0) and (self.isrc is None or self.isrc.tmul != 0):
					tod = d.scan.get_samples()
					# To avoid losing precision, we only reduce precision after subtracting
					# the mean.
					tod-= np.mean(tod,1)[:,None]
					tod = tod.astype(self.dtype)
				else:
					tod = np.zeros([d.scan.ndet,d.scan.nsamp],dtype=self.dtype)
				if self.imap is not None:
					d.pmat_imap.forward(tod, self.imap.map.work[d.sub], tmul=self.imap.tmul, mmul=self.imap.mmul)
					utils.deslope(tod, inplace=True)
				if self.isrc is not None:
					d.pmat_isrc.forward(tod, self.isrc.model.params)
				# Optional azimuth filter
				if self.azfilter is not None:
					todfilter.filter_poly_jon(tod, d.scan.boresight[:,1], naz=self.azfilter.naz, nt=self.azfilter.nt, deslope=True)
			with bench.mark("meq_b_N"):
				d.nmat.apply(tod)
			with bench.mark("meq_b_P'"):
				d.pmap.backward(tod,rhs_map.work[d.sub])
				if self.azmap: d.pmat_azmap.backward(tod, rhs_azmap[azdi])
				d.pcut.backward(tod,rhs_junk[d.cutrange[0]:d.cutrange[1]])
			del tod
			times = [bench.stats[s]["time"].last for s in ["meq_b_get","meq_b_N","meq_b_P'"]]
			L.debug("meq b get %5.1f N %5.3f P' %5.3f" % tuple(times))
		with bench.mark("meq_b_red"):
			rhs_map.work2tile()
			if self.azmap and self.azmap.shared:
				rhs_azmap = reduce(rhs_azmap, self.comm)
		if self.azmap:
			return rhs_map, rhs_junk, rhs_azmap
		else:
			return rhs_map, rhs_junk
	def A(self, map, junk, azmap=None, white=False):
		map, junk = map.copy(), junk.copy()
		omap, ojunk = map.copy().fill(0), junk*0
		# Project map tiles down to local workspaces
		map.tile2work()
		if self.azmap:
			azmap = azmap.copy()
			oazmap = azmap*0
		for di, d in enumerate(self.data):
			azdi = 0 if azmap is not None and len(azmap) <= 1 else di
			with bench.mark("meq_A_P"):
				tod = np.zeros([d.scan.ndet,d.scan.nsamp],dtype=self.dtype)
				d.pmap.forward(tod,map.work[d.sub])
				if self.azmap: d.pmat_azmap.forward(tod,azmap[azdi])
				d.pcut.forward(tod,junk[d.cutrange[0]:d.cutrange[1]])
			with bench.mark("meq_A_N"):
				if white:
					d.nmat.white(tod)
				else:
					d.nmat.apply(tod)
			with bench.mark("meq_A_P'"):
				d.pcut.backward(tod,ojunk[d.cutrange[0]:d.cutrange[1]])
				if self.azmap: d.pmat_azmap.backward(tod,oazmap[azdi])
				d.pmap.backward(tod,omap.work[d.sub])
			del tod
			times = [bench.stats[s]["time"].last for s in ["meq_A_P","meq_A_N","meq_A_P'"]]
			L.debug("meq A P %5.3f N %5.3f P' %5.3f" % tuple(times))
		with bench.mark("meq_A_red"):
			omap.work2tile()
			if self.azmap and self.azmap.shared:
				oazmap = reduce(oazmap, self.comm)
		if self.azmap:
			return omap, ojunk, oazmap
		else:
			return omap, ojunk
	def white(self, map, junk, azmap=None):
		return self.A(map, junk, azmap=azmap, white=True)
	def hitcount(self):
		hitmap = self.area.copy()
		junk   = np.zeros(self.njunk, self.dtype)
		for d in self.data:
			tod = np.full([d.scan.ndet,d.scan.nsamp],1,dtype=self.dtype)
			d.pcut.backward(tod,junk)
			d.pmap.backward(tod,hitmap.work[d.sub])
		hitmap = hitmap[0].astype(np.int32)
		hitmap.work2tile()
		return hitmap
	def postprocess(self, map, div):
		"""Prepare map for output. Add back things that have been temporarily
		subtracted, and finish any pending filtering operations."""
		omap = map.copy()
		if self.azfilter:
			map.tile2work()
			omap.fill(0)
			rhs_junk = np.zeros(self.njunk, dtype=self.dtype)
			for di, d in enumerate(self.data):
				tod = np.zeros([d.scan.ndet,d.scan.nsamp],dtype=self.dtype)
				d.pmap.forward(tod,map.work[d.sub])
				todfilter.filter_poly_jon(tod, d.scan.boresight[:,1], naz=self.azfilter.naz, nt=self.azfilter.nt, deslope=False)
				d.nmat.white(tod)
				# We don't care about cuts, but there were used in div, so we must remove them here too
				d.pcut.backward(tod,rhs_junk)
				d.pmap.backward(tod,omap.work[d.sub])
				del tod
			omap.work2tile()
			for otile, dtile in zip(omap.tiles, div.tiles):
				otile[...] = array_ops.solve_masked(dtile, otile, [0,1])
		if self.isrc and self.isrc.tmul != 0:
			for otile, stile in zip(omap.tiles, self.isrc.map.tiles):
				otile += stile
			#omap += self.isrc.model.draw(map.shape, map.wcs, window=True)
		return omap

class PrecondBinned:
	"""This class implements a simple "binned" preconditioner, which
	disregards detector and time correlations, and solves the system on
	a pixel by pixel basis. It does take into account correlations between
	the different signal components inside each pixel, though."""
	def __init__(self, mapeq):
		ncomp     = mapeq.area.shape[0]
		# Compute the per pixel approximate inverse covmat. Constructing a dmap based on an existing
		# one needs to be simplified! 6 arguments all the time is too much.
		div_map   = dmap.Dmap((ncomp,ncomp)+mapeq.area.shape[-2:],mapeq.area.wcs, mapeq.area.bbpix, tshape=mapeq.area.tshape, comm=mapeq.area.comm, dtype=mapeq.area.dtype)
		div_junk  = np.zeros(mapeq.njunk, dtype=mapeq.area.dtype)
		if mapeq.azmap:
			div_azmap = np.zeros((ncomp,)+mapeq.azshape, dtype=mapeq.area.dtype)
		for ci in range(ncomp):
			for dtile in div_map.tiles: dtile[ci,ci] = 1
			div_junk[...]  = 1
			# Don't set azmap to one here. We get our one value from map.
			# All these special cases are very ugly, and will be handled better
			# in mapmaking.py later.
			if mapeq.azmap:
				div_map[ci], div_junk, div_azmap[ci] = mapeq.white(div_map[ci], div_junk, div_azmap[ci])
			else:
				div_map[ci], div_junk = mapeq.white(div_map[ci], div_junk)
		# Make sure we're symmetric in the TQU-direction
		for dtile in div_map.tiles:
			dtile[:] = 0.5*(dtile+np.rollaxis(dtile,1))
		self.div_map, self.div_junk = div_map, div_junk
		self.hitmap = mapeq.hitcount()
		self.mapeq  = mapeq
		# Compute the pixel component masks, and use it to mask out the
		# corresonding parts of the map preconditioner
		self.mask = makemask(self.div_map)
		for dtile, mtile in zip(self.div_map.tiles, self.mask.tiles):
			dtile *= mtile[None,:]*mtile[:,None]
		if mapeq.azmap:
			# Reshape div_azmap to sensible shape
			div_azmap = np.rollaxis(div_azmap, 2)
			div_azmap = 0.5*(div_azmap+np.rollaxis(div_azmap,1))
			azmask = makemask(div_azmap)
			div_azmap *= azmask[None,:]*azmask[:,None]
			self.div_azmap = np.rollaxis(div_azmap, 2)
	def apply(self, map, junk, azmap=None):
		with bench.mark("prec_bin"):
			rmap = map.copy()
			for rtile, dtile, mtile in zip(rmap.tiles, self.div_map.tiles, map.tiles):
				rtile[:] = array_ops.solve_masked(dtile, mtile, [0,1])
			if azmap is not None:
				res = rmap, junk/self.div_junk, array_ops.solve_masked(self.div_azmap, azmap, [1,2])
			else:
				res = rmap, junk/self.div_junk
		return res
	def write(self, prefix="", ext="fits"):
		dmap.write_map(prefix + "div." + ext, self.div_map)
		dmap.write_map(prefix + "hits." + ext, self.hitmap)
		dmap.write_map(prefix + "mask." + ext, self.mask.astype(np.uint8))

config.default("precon_cyc_npoint", 1, "Number of points to sample in cyclic preconditioner.")
class PrecondCirculant:
	"""This preconditioner approximates the A matrix as
	SCS, where S is a position-dependent standard deviation,
	and C is a position-independent correlation pattern.
	It works well for maps with uniform scanning patterns."""
	def __init__(self, mapeq):
		raise NotImplementedError("PrecondCirculant needs to be adapted to distributed maps")
		ncomp, h,w = mapeq.area.shape
		binned = PrecondBinned(mapeq)

		S  = array_ops.eigpow(binned.div_map, -0.5, axes=[0,1])

		# Sample 4 points to avoid any pathologies
		#N  = 2
		#pix = [[h*(2*i+1)/N/2,w*(2*j+1)/N/2] for i in range(N) for j in range(0,N)]
		#pix = np.array([[-1,-1],[1,1]])*10+np.array([h/2,w/2])[None,:]
		npoint = config.get("precon_cyc_npoint")
		pix = pick_ref_points(binned.div_map[0,0], npoint)
		Arow = measure_corr_cyclic(mapeq, S, pix)
		# Measure this fft, since we will perform it a lot
		#fft.ifft(fft.fft(Arow.copy(), axes=[-2,-1], flags=["FFTW_MEASURE"]),axes=[-2,-1], flags=["FFTW_MEASURE"])
		iC = fft.fft(Arow, axes=[-2,-1])
		C  = enmap.samewcs(array_ops.eigpow(iC,-1,axes=[0,1]), binned.div_map)

		self.Arow = enmap.samewcs(Arow, binned.div_map)
		self.S, self.C = S, C
		self.div_junk = binned.div_junk
		self.div_map  = binned.div_map
		self.mask = binned.mask
		self.binned = binned
		self.mapeq = mapeq
	def apply(self, map, junk):
		# We will apply the operation m \approx S C S map
		# The fft normalization is baked into iC.
		with bench.mark("prec_cyc"):
			m  = enmap.map_mul(self.S, map)
			mf = fft.fft(m, axes=[-2,-1])
			mf = enmap.map_mul(self.C, mf)
			m  = fft.ifft(mf, axes=[-2,-1], normalize=True).real
			m  = enmap.map_mul(self.S, m)
		return m, junk/self.div_junk
	def write(self, prefix="", ext="fits"):
		if self.mapeq.comm.rank > 0: return
		enmap.write_map(prefix + "arow." + ext, self.Arow)
		self.binned.write(prefix, ext=ext)

def pick_ref_points(hitmap, npoint):
	pix = []
	w   = hitmap.copy()
	# kill edge, since hits may accumulate there
	w[0]=0;w[-1]=0;w[:,0]=0;w[:,-1]=0
	# Bias us towards center of map
	com = np.sum(w.pixmap()*hitmap,(-2,-1))/np.sum(hitmap)
	dist= np.sum((w.pixmap()-com[:,None,None])**2,0)**0.5
	w  *= np.exp(-4*dist**2/np.product(w.shape[-2:]))
	# Find typical radius of hitmap
	area_tot  = np.sum(w)/np.max(w)
	area_mask = area_tot/npoint/3
	r_mask    = (area_mask/np.pi)**0.5
	for i in range(npoint):
		# Find highest-weight point
		pix.append(np.unravel_index(np.argmax(w),w.shape))
		# Mask surrounding area
		mask = np.zeros(hitmap.shape)+1
		mask[tuple(pix[-1])] = 0
		mask = ndimage.distance_transform_edt(mask)>r_mask
		w *= mask
	return np.array(pix)+1

config.default("precond_condition_lim", 10., "Maximum allowed condition number in per-pixel polarization matrices.")
def makemask(div):
	masks = div[0].copy().astype(bool)
	lim  = config.get("precond_condition_lim")
	for dtile, mtile in zip(div.tiles, masks.tiles):
		condition = array_ops.condition_number_multi(dtile, [0,1])
		tmask = dtile[0,0] > 0
		pmask = (condition >= 1)*(condition < lim)
		mtile[0]  = tmask
		mtile[1:] = pmask[None]
		del condition
	return masks

def reduce(a, comm=MPI.COMM_WORLD):
	res = a.copy()
	comm.Allreduce(a, res)
	return res

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
		Arow = enmap.map_mul(Arow, S)
		Arow = enmap.map_mul(Sref, Arow)
		Arow = np.roll(Arow, -p[0], 2)
		Arow = np.roll(Arow, -p[1], 3)
		d += Arow
	d /= len(pixels)
	# We should be symmetric from the beginning, but it turns out
	# we're not. Should investigate that. In the mean while,
	# symmetrize so that conjugate gradients doesn't break down.
	return d
	#return sympos(d)

def sympos(arow):
	f = fft.fft(arow, axes=[-2,-1])
	print "sympos A", np.min(f.real), np.min(f.imag), np.max(f.real), np.max(f.imag)
	# Make us symmetric in real space by killing the imaginary part
	f = f.real
	# Make us symmetric in component space
	f = 0.5*(f+np.rollaxis(f,1))
	# Remove negative eigenvalues
	f = array_ops.eigflip(f, axes=[0,1])
	x = fft.ifft(f+0j, axes=[-2,-1], normalize=True).real
	return x


def normalize(A):
	# Normalize to unit diagonal
	D = np.maximum(1e-30,np.abs(np.diag(A)))**-0.5
	A = A * D[:,None]
	A *= D[None,:]
	return A

def checksym(A):
	A = normalize(A)
	n = len(A)
	res = np.zeros(n)
	for i in range(n):
		res[i] = np.max(np.abs(A[:,i]-A[i,:]))
	return res

def checkeig(A):
	A = normalize(A)
	e,v = np.linalg.eig(A)
	return e/np.max(e)

def test_symmetry(mapeq, nmax=0, shuf=True, verbose=True, prec=None):
	# Measure the typical correlation pattern by using multiple
	# pixels at the same time.
	mask = mapeq.area.astype(bool)+True
	dof  = DOF(Arg(mask=mask),Arg(shape=(mapeq.njunk,),distributed=True))
	a = np.random.standard_normal(dof.n).astype(mapeq.dtype)
	mask = mapeq.A(*dof.unzip(a))[0] != 0
	dof  = DOF(Arg(mask=mask),Arg(shape=(mapeq.njunk,),distributed=True))
	fun = mapeq.A
	if prec:
		def fun(*args): return prec.apply(*mapeq.A(*mapeq.A(*prec.apply(*args))))
		#def fun(*args): return prec.apply(*args)
	if nmax == 0: nmax = dof.n-mapeq.njunk
	rows = []
	if shuf:
		inds = np.random.permutation(dof.n-mapeq.njunk)[:nmax]
		inds = mapeq.comm.bcast(inds)
	else:
		inds = np.arange(nmax)
	for i, ind in enumerate(inds):
		a = np.zeros(dof.n,dtype=mapeq.dtype); a[ind] = 1
		rows.append(dof.zip(*fun(*dof.unzip(a))))
		if verbose:
			b = np.array(rows)[:,np.array(inds[:i+1])]
			sym = checksym(b)
			eig = checkeig(b)
			print "sym %4d %15.7e %15.7e" % (i, np.max(sym), np.min(eig))
	if mapeq.comm.rank == 0:
		A = np.array(rows)[:,inds[:nmax]]
		with h5py.File("A.hdf","w") as hfile:
			hfile["data"] = A




# Submap preconditioner stuff below here. I never got this to work,
# but it's still an interesting idea.
# =================================================================

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
	def __init__(self, mapeq, precon="bin"):
		binned = PrecondBinned(mapeq)

		# Categorize each scan into groups which can be
		# combined into one large scan with the same
		# scanning pattern as the individual scans.
		scaninfo = []
		for d in mapeq.data:
			scaninfo.append(analyze_scan(d))
		allinfo = mapeq.comm.allreduce(scaninfo)
		#extra = copy.deepcopy(allinfo[0])
		#estep = 40000
		#extra.ibox += extra.ivecs[1]*estep
		#extra.obox += extra.ovecs[1]*estep
		#allinfo.append(extra)
		groups  = group_scans_by_scandirs(allinfo)
		groups  = split_disjoint_scan_groups(groups)
		# Assign ownership of groups
		mygroups = groups[mapeq.comm.rank::mapeq.comm.size]
		# For each group, define a single effective scan
		myscans = [sim_scan_from_group(group, mapeq.area) for group in mygroups]
		self.linsys = LinearSystemMap(myscans, mapeq.area, mapeq.comm, precon=precon)
		self.nmax = 20
		self.mask = binned.mask

	def apply(self, map, junk):
		eq     = self.linsys
		b      = eq.dof.zip(map,junk)
		solver = cg.CG(eq.A, b, M=eq.M, dot=eq.dof.dot)
		for i in range(self.nmax):
			t1 = time.time()
			solver.step()
			t2 = time.time()
			print "sub %5d %15.7e %6.3f" % (solver.i, solver.err, t2-t1)
			#map, _ = eq.dof.unzip(solver.x)
			#enmap.write_map("sub%03d.hdf" % solver.i, map)
		map, _ = eq.dof.unzip(solver.x)
		return map, junk
	def write(self, prefix="", ext="fits"):
		self.binned.write(prefix, ext=ext)

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

def analyze_scan(d):
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

	return bunch.Bunch(
			ibox   = box,   obox  = obox,
			ivecs  = ivecs, ovecs = ovecs,
			sys    = d.scan.sys,
			site   = d.scan.site,
			mjd0   = d.scan.mjd0,
			noise  = d.nmat,
			offsets= d.scan.offsets,
			comps  = d.scan.comps,
			ncomp  = len(d.pmap.comps),
			scandim=dscan)

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
			posdiff = utils.decomp_basis(me.ovecs, me.obox-other.obox)
			posdiff = np.sum(posdiff**2,0)**0.5
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
		# Compute from-to for each in the drift direction
		driftvec = group[0].ovecs[1]
		driftvec /= np.sum(driftvec**2)**0.5
		pos = np.array([np.sum(member.obox*driftvec,1) for member in group])
		# Sort group by the starting position
		inds = np.argsort(pos[:,0])
		group, pos = [group[i] for i in inds], pos[inds]
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

def build_triangle_wave(ibox, ivec):
	"""Build a triangle wave scan based on the given bounding box and
	scan/drift per-sample-steps ivec. Returns an [nsamp,ncoord] array."""
	period, nsamp = utils.decomp_basis(ivec, ibox[1]-ibox[0])
	phase = np.arange(nsamp)%(2*period)
	phase[phase>period] = 2*period-phase[phase>period]
	return (ibox[0,:,None] + ivec[1][:,None]*np.arange(nsamp)[None,:] + ivec[0][:,None]*phase[None,:]).T

def sim_scan_from_group(group, area, oversample=2):
	obox_tot = np.array([np.min([g.obox[0] for g in group],0),np.max([g.obox[1] for g in group],0)])
	# We already know that the scanning directions and amplitudes agree.
	# So to construct the total bounds in input coordinates, we just need
	# to take into account the drift vector. First compute the number of drift
	# vectors we are off. out_off = ovec*x, in_off = ivec*x,
	# x = (ovec'ovec)"ovec'out_off
	ivec, iref = group[0].ivecs, group[0].ibox
	ovec, oref = group[0].ovecs, group[0].obox

	# We will oversample by a certain factor to ensure that we hit every
	# pixel. This means that the simulated telescope moves less per sample,
	# and that the last frequency bin must be extended.
	ivec /= oversample
	ovec /= oversample

	ibox_tot = iref + utils.decomp_basis(ovec, obox_tot-oref).dot(ivec)
	# We must now create new effective scan objects. These require:
	#  boresight: fit to ibox_tot, repeating scanning pattern as necessary
	#  we use a simplified triangle wave scanning pattern.
	bore  = build_triangle_wave(ibox_tot, ivec)
	#  offsets:   ncomp detectors, all with same (zero?) offsets
	ncomp  = group[0].ncomp
	nsamp, ncoord = bore.shape
	ndet   = ncomp
	offset = np.zeros([ndet,ncoord])
	comps  = np.eye(ncomp)                   # only a single comp each
	cut    = rangelist.zeros([ndet,nsamp])   # no cuts
	sys, site, mjd0 = group[0].sys, group[0].site, group[0].mjd0

	noise = build_effective_noise_model(group, nsamp)
	noise.bins[-1,-1] *= oversample

	return scansim.SimMap(map=area, boresight=bore, offsets=offset, comps=comps, cut=cut, sys=sys, site=site, mjd0=mjd0, noise=noise)

def build_effective_noise_model(group, nsamp):
	# Next, construct a noise model for this scan. If all the individual
	# scans perfectly overlapped, this would simply be the sum of all
	# the inverse covariances projected down to our ncomp detectors.
	# Because they don't all overlap, we first need to renormalize
	# the noise in tod space, but that doesn't fit into our current
	# noise model.
	#
	# Basically, we need
	#  Ntot(x) = N(x/norm)*norm
	# except that we need to handle norm ~ 0 or norm == None
	# Norm would be a per-sample mask, and would be quite large under
	# normal circumstances, though not in our case. Since this is
	# so simple, it might be best to include this as an optional
	# element in NmatDetvecs and the normal noise parameters.
	# For now, just compute what we would get if they did overlap.
	ncomp     = group[0].ncomp
	noise_tot = copy.deepcopy(group[0].noise)
	# The individual scans may not have the same frequency bins.
	# We handle this by rebinning to the bins of the first member.
	bins     = group[0].noise.bins
	nbin     = bins.shape[0]

	iC  = np.zeros([nbin,ncomp,ncomp])
	for member in group:
		noise    = member.noise
		# Compute mapping between local and global binning
		bcenters = np.mean(bins,1)
		inds     = np.searchsorted(noise.bins[:,1], bcenters)
		inds     = inds + (bcenters-noise.bins[inds,0])/(noise.bins[inds,1]-noise.bins[inds,0])

		# Compute detector-collapsed local noise parameters, assuming
		# equal noise for each
		comps    = member.comps[:,:ncomp]
		small_iC = np.zeros([len(noise.bins),ncomp,ncomp])
		for i,icov in enumerate(noise.icovs):
			small_iC[i] = comps.T.dot(icov).dot(comps)
		# Interpolate to global bins
		iC  += utils.interpol(small_iC.T,  inds[None,:], order=1).T

	# Build a dense binned noise model based on this
	return nmat.NmatBinned(iC, bins)

