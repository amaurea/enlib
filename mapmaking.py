# Separable map making. Defines a set of Signals, which contain forward
# and backaward projection as well as a preconditioner. These can then be
# used via
#
# A matrix (cut is one of the signals here)
# for each scan
#  for each (signal,sflat)
#   signal.forward(scan,tod,sflat)
#  scan.noise.apply(tod)
#  for each (signal,sflat) reverse
#   signal.backward(scan,tod,sflat)
#
# M matrix
# for each (signal,sflat)
#  signal.precon(sflat)
#
# Init: Because we want to treat cuts as a signal, but also want cuts
# to be available for other signals to use, this is a bit cumbersome.
# We make sure to initialize the cut signal first, and pass it as an
# extra argument to the other signals
# for each scan
#  signal_cut.add(scan)
#  for each other signal
#   signal.add(scan, signal_cut)
#
# Or perhaps one should add the cut signal reference when *initializing*
# the signal objects, and then require cuts to be passed first in the
# signal array (in general passing signals other signals depend on first).
# Then it would look like
#
# for each scan
#  for each signal
#   signal.add(scan)
#
# signal_cut   = SignalCut(...)
# signal_map   = SignalMap(..., cut=signal_cut)
# signal_phase = SignalPhase(..., cut=signal_cut)
# signals = [signal_cut, signal_map, signal_phase]
import numpy as np, h5py, zipper, logging, gc
from . import enmap, dmap, array_ops, pmat, utils, todfilter
from . import config, nmat, bench, gapfill, mpi, sampcut, fft
from .cg import CG
L = logging.getLogger(__name__)

def dump(fname, d):
	print "dumping " + fname
	with h5py.File(fname, "w") as hfile:
		hfile["data"] = d

######## Signals ########

class Signal:
	"""General signal interface."""
	def __init__(self, name, ofmt, output, ext):
		self.ext    = ext
		self.dof    = zipper.ArrayZipper(np.zeros([0]))
		self.precon = PreconNull()
		self.prior  = PriorNull()
		self.post   = []
		self.name   = name
		self.ofmt   = ofmt
		self.output = output
	def prepare (self, x): return x.copy()
	def forward (self, scan, tod, x): pass
	def backward(self, scan, tod, x): pass
	def precompute(self, scan): pass
	def free(self): pass
	def finish  (self, x, y): x[:] = y
	# This one is a potentially cheaper version of self.prepare(self.zeros())
	def work    (self): return self.prepare(self.zeros())
	def write   (self, prefix, tag, x): pass
	def postprocess(self, x):
		for p in self.post: x = p(x)
		return x

class SignalMap(Signal):
	def __init__(self, scans, area, comm, cuts=None, name="main", ofmt="{name}", output=True,
			ext="fits", pmat_order=None, sys=None, nuisance=False, data=None):
		Signal.__init__(self, name, ofmt, output, ext)
		self.area = area
		self.cuts = cuts
		self.dof  = zipper.ArrayZipper(area, comm=comm)
		self.dtype= area.dtype
		if data is not None:
			self.data = data
		else:
			self.data = {scan: pmat.PmatMap(scan, area, order=pmat_order, sys=sys) for scan in scans}
	def forward(self, scan, tod, work):
		if scan not in self.data: return
		self.data[scan].forward(tod, work)
	def backward(self, scan, tod, work):
		if scan not in self.data: return
		self.data[scan].backward(tod, work)
	def finish(self, m, work):
		self.dof.comm.Allreduce(work, m)
	def zeros(self): return enmap.zeros(self.area.shape, self.area.wcs, self.area.dtype)
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		if self.dof.comm.rank == 0:
			enmap.write_map(oname, m)

class SignalMapFast(SignalMap):
	def __init__(self, scans, area, comm, cuts=None, name="main", ofmt="{name}", output=True,
			ext="fits", sys=None, nuisance=False, data=None):
		if data is None:
			data = {scan: pmat.PmatMapFast(scan, area, sys=sys) for scan in scans}
		SignalMap.__init__(self, scans, area, comm=comm, cuts=cuts, name=name, ofmt=ofmt,
				output=output, ext=ext, sys=sys, nuisance=nuisance, data=data)
	def precompute(self, scan):
		if scan not in self.data: return
		self.pix, self.phase = self.data[scan].get_pix_phase()
	def free(self):
		del self.pix, self.phase
	def forward(self, scan, tod, work):
		if scan not in self.data: return
		self.data[scan].forward(tod, work, self.pix, self.phase)
	def backward(self, scan, tod, work):
		if scan not in self.data: return
		self.data[scan].backward(tod, work, self.pix, self.phase)

config.default("dmap_format","merged","How to store dmaps on disk. 'merged': combine into a single fits file before writing. This is memory intensive. 'tiles': Write the tiles that make up the dmap directly to disk.")
class SignalDmap(Signal):
	def __init__(self, scans, subinds, area, cuts=None, name="main", ofmt="{name}", output=True,
			ext="fits", pmat_order=None, sys=None, nuisance=False, data=None):
		Signal.__init__(self, name, ofmt, output, ext)
		self.area = area
		self.cuts = cuts
		self.dof  = dmap.DmapZipper(area)
		self.dtype= area.dtype
		self.subs = subinds
		if data is None:
			data = {}
			work = area.tile2work()
			for scan, subind in zip(scans, subinds):
				data[scan] = [pmat.PmatMap(scan, work[subind], order=pmat_order, sys=sys), subind]
		self.data = data
	def prepare(self, m):
		return m.tile2work()
	def forward(self, scan, tod, work):
		if scan not in self.data: return
		mat, ind = self.data[scan]
		mat.forward(tod, work[ind])
	def backward(self, scan, tod, work):
		if scan not in self.data: return
		mat, ind = self.data[scan]
		mat.backward(tod, work[ind])
	def finish(self, m, work):
		m.work2tile(work)
	def zeros(self): return dmap.zeros(self.area.geometry)
	def work(self):  return self.area.geometry.build_work()
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		merged= config.get("dmap_format") == "merged"
		dmap.write_map(oname, m, merged=merged)

class SignalDmapFast(SignalDmap):
	def __init__(self, scans, subinds, area, cuts=None, name="main", ofmt="{name}", output=True,
			ext="fits", sys=None, nuisance=False, data=None):
		if data is None:
			data = {}
			work = area.tile2work()
			for scan, subind in zip(scans, subinds):
				data[scan] = [pmat.PmatMapFast(scan, work[subind], sys=sys), subind]
		SignalDmap.__init__(self, scans, subinds, area, cuts=cuts, name=name, ofmt=ofmt,
				output=output, ext=ext, sys=sys, nuisance=nuisance, data=data)
	def precompute(self, scan):
		if scan not in self.data: return
		mat, ind = self.data[scan]
		self.pix, self.phase = mat.get_pix_phase()
	def free(self):
		del self.pix, self.phase
	def forward(self, scan, tod, work):
		if scan not in self.data: return
		mat, ind = self.data[scan]
		mat.forward(tod, work[ind], self.pix, self.phase)
	def backward(self, scan, tod, work):
		if scan not in self.data: return
		mat, ind = self.data[scan]
		mat.backward(tod, work[ind], self.pix, self.phase)

class SignalCut(Signal):
	def __init__(self, scans, dtype, comm, name="cut", ofmt="{name}_{rank:02}", output=False, cut_type=None):
		Signal.__init__(self, name, ofmt, output, ext="hdf")
		self.data  = {}
		self.dtype = dtype
		cutrange = [0,0]
		for scan in scans:
			mat = pmat.PmatCut(scan, cut_type)
			cutrange = [cutrange[1], cutrange[1]+mat.njunk]
			self.data[scan] = [mat, cutrange]
		self.njunk = cutrange[1]
		self.dof = zipper.ArrayZipper(np.zeros(self.njunk, self.dtype), shared=False, comm=comm)
	def forward(self, scan, tod, junk):
		if scan not in self.data: return
		mat, cutrange = self.data[scan]
		mat.forward(tod, junk[cutrange[0]:cutrange[1]])
	def backward(self, scan, tod, junk):
		if scan not in self.data: return
		mat, cutrange = self.data[scan]
		mat.backward(tod, junk[cutrange[0]:cutrange[1]])
	def zeros(self): return np.zeros(self.njunk, self.dtype)
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name, rank=self.dof.comm.rank)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		with h5py.File(oname, "w") as hfile:
			hfile["data"] = m

class SignalPhase(Signal):
	def __init__(self, scans, pids, patterns, array_shape, res, dtype, comm, cuts=None,
			name="phase", ofmt="{name}_{pid:02}_{az0:.0f}_{az1:.0f}_{el:.0f}", output=True,
			ext="fits", col_major=True, hysteresis=True):
		Signal.__init__(self, name, ofmt, output, ext)
		nrow,ncol = array_shape
		ndet = nrow*ncol
		self.pids = pids
		self.patterns = patterns
		self.comm = comm
		self.dtype = dtype
		self.col_major = col_major
		self.cuts = cuts
		self.data = {}
		self.areas = []
		# Set up an area for each scanning pattern. We assume that these are constant
		# elevation scans, so only azimuth matters. This setup is ugly and would be
		# nicer if passed in, but then the ugliness would only be moved to the calling
		# code instead.
		for pattern in patterns:
			az0,az1 = utils.widen_box(pattern)[:,1]
			naz = int(np.ceil((az1-az0)/res))
			az1 = az0 + naz*res
			det_unit = nrow if col_major else ncol
			shape, wcs = enmap.geometry(pos=[[0,az0],[ndet/det_unit*utils.degree,az1]], shape=(ndet,naz), proj="car")
			if hysteresis:
				area = enmap.zeros((2,)+shape, wcs, dtype=dtype)
			else:
				area = enmap.zeros(shape, wcs, dtype=dtype)
			self.areas.append(area)
		for pid, scan in zip(pids,scans):
			dets = scan.dets
			if col_major: dets = utils.transpose_inds(dets, nrow, ncol)
			mat = pmat.PmatScan(scan, self.areas[pid], dets)
			self.data[scan] = [pid, mat]
		self.dof = zipper.MultiZipper([
			zipper.ArrayZipper(area, comm=comm) for area in self.areas],
			comm=comm)
	def prepare(self, ms):
		return [m.copy() for m in ms]
	def forward(self, scan, tod, work):
		if scan not in self.data: return
		pid, mat = self.data[scan]
		mat.forward(tod, work[pid])
	def backward(self, scan, tod, work):
		if scan not in self.data: return
		pid, mat = self.data[scan]
		mat.backward(tod, work[pid])
	def finish(self, ms, work):
		for m, w in zip(ms, work):
			self.dof.comm.Allreduce(w,m)
	def zeros(self):
		return [area*0 for area in self.areas]
	def write(self, prefix, tag, ms):
		if not self.output: return
		for pid, (pattern, m) in enumerate(zip(self.patterns, ms)):
			oname = self.ofmt.format(name=self.name, pid=pid, el=pattern[0,0]/utils.degree, az0=pattern[0,1]/utils.degree, az1=pattern[1,1]/utils.degree)
			oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
			if self.dof.comm.rank == 0:
				enmap.write_map(oname, m)

class SignalMapBuddies(SignalMap):
	def __init__(self, scans, area, comm, cuts=None, name="main", ofmt="{name}", output=True, ext="fits", pmat_order=None, sys=None, nuisance=False):
		Signal.__init__(self, name, ofmt, output, ext)
		self.area = area
		self.cuts = cuts
		self.dof  = zipper.ArrayZipper(area, comm=comm)
		self.dtype= area.dtype
		self.comm = comm
		self.data = {scan: [
			pmat.PmatMap(scan, area, order=pmat_order, sys=sys),
			pmat.PmatMapMultibeam(scan, area, scan.buddy_offs,
				scan.buddy_comps, order=pmat_order, sys=sys)
			] for scan in scans}
	def forward(self, scan, tod, work):
		if scan not in self.data: return
		for pmat in self.data[scan]:
			pmat.forward(tod, work)
	def backward(self, scan, tod, work):
		if scan not in self.data: return
		for pmat in self.data[scan]:
			pmat.backward(tod, work)
	# Ugly hack
	def get_nobuddy(self):
		data = {k:v[0] for k,v in self.data.iteritems()}
		return SignalMap(self.data.keys(), self.area, self.comm, cuts=self.cuts, output=False, data=data)

######## Preconditioners ########
# Preconditioners have a lot of overlap with Eqsys.A. That's not
# really surprising, as their job is to approximate A, but it does
# lead to duplicate code. The preconditioners differ by:
#  1. Only invovling [signal,cut]
#  2. Looping over components of the signal (if it has components)
# We can do this via a loop of the type
#  for each compmap:
#    eqsys.A(compmap)
# e.g. one sends in a reduced equation system to the preconditioner:
#  PreconMapBinned(eqsys)
# instead of
#  PreconMapBinned(signal, signal_cut, scans, weights)
# But th eqsys above would not be the real eqsys. So wouldn't it really be
#  PreconMapBinned(Eqsys(scans, [signal, signal_cut], weights=weights))
# which is pretty long. The internal eqsys use is an implementation
# detail. Pass the arguments we need, and use an eqsys internally if
# that makes things simpler. Eqsyses are cheap to construct.

class PreconNull:
	def __init__(self): pass
	def __call__(self, m): pass
	def write(self, prefix="", ext="fits"): pass

config.default("eig_limit", 1e-6, "Smallest relative eigenvalue to invert in eigenvalue inversion. Ones smaller than this are set to zero.")
class PreconMapBinned:
	def __init__(self, signal, signal_cut, scans, weights, noise=True, hits=True):
		"""Binned preconditioner: (P'W"P)", where W" is a white
		nosie approximation of N". If noise=False, instead computes
		(P'P)". If hits=True, also computes a hitcount map."""
		ncomp = signal.area.shape[0]
		self.div = enmap.zeros((ncomp,)+signal.area.shape, signal.area.wcs, signal.area.dtype)
		calc_div_map(self.div, signal, signal_cut, scans, weights, noise=noise)
		self.idiv = array_ops.svdpow(self.div, -1, axes=[0,1], lim=config.get("eig_limit"))
		#self.idiv[:] = np.eye(3)[:,:,None,None]
		if hits:
			# Build hitcount map too
			self.hits = signal.area.copy()
			self.hits = calc_hits_map(self.hits, signal, signal_cut, scans)
		else: self.hits = None
		self.signal = signal
	def __call__(self, m):
		m[:] = array_ops.matmul(self.idiv, m, axes=[0,1])
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)
		if self.hits is not None:
			self.signal.write(prefix, "hits", self.hits)

class PreconDmapBinned:
	def __init__(self, signal, signal_cut, scans, weights, noise=True, hits=True):
		"""Binned preconditioner: (P'W"P)", where W" is a white
		nosie approximation of N". If noise=False, instead computes
		(P'P)". If hits=True, also computes a hitcount map."""
		geom  = signal.area.geometry.copy()
		geom.pre = (signal.area.shape[0],)+geom.pre
		self.div = dmap.zeros(geom)
		calc_div_map(self.div, signal, signal_cut, scans, weights, noise=noise)
		self.idiv = self.div.copy()
		for dtile in self.idiv.tiles:
			dtile[:] = array_ops.svdpow(dtile, -1, axes=[0,1], lim=config.get("eig_limit"))
		if hits:
			# Build hitcount map too
			self.hits = signal.area.copy()
			self.hits = calc_hits_map(self.hits, signal, signal_cut, scans)
		else: self.hits = None
		self.signal = signal
	def __call__(self, m):
		for idtile, mtile in zip(self.idiv.tiles, m.tiles):
			mtile[:] = array_ops.matmul(idtile, mtile, axes=[0,1])
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)
		if self.hits is not None:
			self.signal.write(prefix, "hits", self.hits)

class PreconMapHitcount:
	def __init__(self, signal, signal_cut, scans):
		"""Hitcount preconditioner: (P'P)_TT."""
		ncomp = signal.area.shape[0]
		self.hits = signal.area.copy()
		self.hits = calc_hits_map(self.hits, signal, signal_cut, scans)
		self.ihits= 1.0/np.maximum(self.hits,1)
		self.signal = signal
	def __call__(self, m):
		m *= self.ihits
		#hits = np.maximum(self.hits, 1)
		#m /= hits
	def write(self, prefix):
		self.signal.write(prefix, "hits", self.hits)

class PreconDmapHitcount:
	def __init__(self, signal, signal_cut, scans):
		"""Hitcount preconditioner: (P'P)_TT"""
		geom  = signal.area.geometry.copy()
		geom.pre = (signal.area.shape[0],)+geom.pre
		self.hits = signal.area.copy()
		self.hits = calc_hits_map(self.hits, signal, signal_cut, scans)
		self.signal = signal
	def __call__(self, m):
		for htile, mtile in zip(self.hits.tiles, m.tiles):
			htile = np.maximum(htile, 1)
			mtile /= htile
	def write(self, prefix):
		self.signal.write(prefix, "hits", self.hits)

class PreconCut:
	def __init__(self, signal, scans):
		junk  = signal.zeros()
		iwork = signal.prepare(junk)
		owork = signal.prepare(junk)
		iwork[:] = 1
		for scan in scans:
			with bench.mark("div_" + signal.name):
				tod = np.zeros((scan.ndet, scan.nsamp), iwork.dtype)
				signal.forward (scan, tod, iwork)
				scan.noise.white(tod)
				signal.backward(scan, tod, owork)
			times = [bench.stats[s]["time"].last for s in ["div_"+signal.name]]
			L.debug("div %s %6.3f %s" % ((signal.name,)+tuple(times)+(scan.id,)))
		signal.finish(junk, owork)
		self.idiv = junk*0
		self.idiv[junk!=0] = 1/junk[junk!=0]
		self.signal = signal
	def __call__(self, m):
		m *= self.idiv
	def write(self, prefix):
		self.signal.write(prefix, "idiv", self.idiv)

class PreconPhaseBinned:
	def __init__(self, signal, signal_cut, scans, weights):
		div = [area*0+1 for area in signal.areas]
		# The cut samples are included here becuase they must be avoided, but the
		# actual computation of the junk sample preconditioner happens elsewhere.
		# This is a bit redundant, but should not cost much time since this only happens
		# in the beginning.
		ijunk = np.zeros(signal_cut.njunk, dtype=signal.dtype)
		ojunk = signal_cut.prepare(signal_cut.zeros())
		iwork = signal.prepare(div)
		owork = signal.prepare(signal.zeros())
		prec_div_helper(signal, signal_cut, scans, weights, iwork, owork, ijunk, ojunk)
		signal.finish(div, owork)
		hits = signal.zeros()
		owork = signal.prepare(hits)
		for scan in scans:
			tod = np.full((scan.ndet, scan.nsamp), 1, signal.dtype)
			signal_cut.backward(scan, tod, ojunk)
			signal.backward(scan, tod, owork)
		signal.finish(hits, owork)
		for i in range(len(hits)):
			hits[i] = hits[i].astype(np.int32)
		self.div = div
		self.hits = hits
		self.signal = signal
	def __call__(self, ms):
		for d, m in zip(self.div, ms):
			m[d!=0] /= d[d!=0]
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)

def prec_div_helper(signal, signal_cut, scans, weights, iwork, owork, ijunk, ojunk, noise=True):
	# The argument list of this one is so long that it almost doesn't save any
	# code.
	for scan in scans:
		with bench.mark("div_Pr_" + signal.name):
			signal.precompute(scan)
		with bench.mark("div_P_" + signal.name):
			tod = np.zeros((scan.ndet, scan.nsamp), signal.dtype)
			signal.forward(scan, tod, iwork)
			signal_cut.forward (scan, tod, ijunk)
		with bench.mark("div_white"):
			for weight in weights: weight(scan, tod)
			if noise: scan.noise.white(tod)
			for weight in weights[::-1]: weight(scan, tod)
		with bench.mark("div_PT_" + signal.name):
			signal_cut.backward(scan, tod, ojunk)
			signal.backward(scan, tod, owork)
		with bench.mark("div_Fr_" + signal.name):
			signal.free()
		times = [bench.stats[s]["time"].last for s in ["div_P_"+signal.name, "div_white", "div_PT_" + signal.name]]
		L.debug("div %s %6.3f %6.3f %6.3f %s" % ((signal.name,)+tuple(times)+(scan.id,)))

def calc_div_map(div, signal, signal_cut, scans, weights, noise=True):
	# The cut samples are included here becuase they must be avoided, but the
	# actual computation of the junk sample preconditioner happens elsewhere.
	# This is a bit redundant, but should not cost much time since this only happens
	# in the beginning.
	ijunk= np.zeros(signal_cut.njunk, dtype=signal.area.dtype)
	ojunk= signal_cut.prepare(signal_cut.zeros())
	for i in range(div.shape[0]):
		div[i,i] = 1
		iwork = signal.prepare(div[i])
		owork = signal.prepare(signal.zeros())
		prec_div_helper(signal, signal_cut, scans, weights, iwork, owork, ijunk, ojunk, noise=noise)
		signal.finish(div[i], owork)

def calc_crosslink_map(cmap, signal, signal_cut, scans, weights, noise=True):
	# This is really messy, but it should work. The signal contain a list of
	# pmats, each of which has a reference to a scan which contains the polarization
	# phase information. Since it's just a reference, we can change the scans instead
	# of mucking around inside the pmats. But this is a fragile construction - if
	# pmat internals change, it will break.
	saved_comps = [scan.comps.copy() for scan in scans]
	for scan in scans: scan.comps[:] = np.array([1,1,0])
	calc_div_map(cmap, signal, signal_cut, scans, weights, noise=noise)
	for scan, comp in zip(scans, saved_comps): scan.comps = comp

def calc_hits_map(hits, signal, signal_cut, scans):
	work = signal.prepare(hits)
	ojunk= signal_cut.prepare(signal_cut.zeros())
	for scan in scans:
		with bench.mark("hits_Pr_" + signal.name):
			signal.precompute(scan)
		with bench.mark("hits_PT"):
			tod = np.full((scan.ndet, scan.nsamp), 1, hits.dtype)
			signal_cut.backward(scan, tod, ojunk)
			signal.backward(scan, tod, work)
		with bench.mark("hits_Fr_" + signal.name):
			signal.free()
		times = [bench.stats[s]["time"].last for s in ["hits_PT"]]
		L.debug("hits %s %6.3f %s" % ((signal.name,)+tuple(times)+(scan.id,)))
	with bench.mark("hits_reduce"):
		signal.finish(hits, work)
	return hits[0].astype(np.int32)

######## Priors ########

class PriorNull:
	def __call__(self, scans, xin, xout): pass

class PriorNorm:
	def __init__(self, epsilon=1e-3):
		self.epsilon = epsilon
	def __call__(self, scans, imap, omap):
		omap += self.epsilon * imap

class PriorMapNohor:
	def __init__(self, weight=1):
		self.weight = weight
	def __call__(self, scans, imap, omap):
		omap += np.asarray(np.sum(imap*self.weight,-1))[:,:,None]*self.weight

class PriorDmapNohor:
	def __init__(self, weight=1):
		self.weight = weight
	def __call__(self, scans, imap, omap):
		tmp = omap*0
		dmap.broadcast_into(tmp, dmap.sum(imap*self.weight,-1), -1)
		omap += tmp * self.weight

class PriorProjectOut:
	def __init__(self, signal_map, signal_pickup, weight=1):
		self.signal_map = signal_map
		self.signal_pickup = signal_pickup
		self.weight = weight
	def __call__(self, scans, imap, omap):
		pmap  = self.signal_pickup.zeros()
		mmap  = self.signal_map.zeros()
		mwork = self.signal_map.prepare(self.signal_map.zeros())
		pwork = self.signal_pickup.prepare(pmap)
		for scan in scans:
			tod = np.zeros([scan.ndet, scan.nsamp], self.signal_map.dtype)
			self.signal_map.forward(scan, tod, mwork)
			self.signal_pickup.backward(scan, tod, pwork)
		self.signal_pickup.finish(pmap, pwork)
		for pm, h in zip(pmap,self.signal_pickup.precon.hits):
			pm /= h*h
		mwork = self.signal_map.prepare(mmap)
		for scan in scans:
			tod = np.zeros([scan.ndet, scan.nsamp], self.signal_map.dtype)
			self.signal_pickup.forward(scan, tod, pwork)
			self.signal_map.backward(scan, tod, mwork)
		self.signal_map.finish(mmap, mwork)
		omap += mmap*self.weight

######## Filters ########
# Possible filters: A (P'BN"BP)" P'BN"BC d
# B: weighting filter: windowing
# C: preprocessing filter:  source subtraction, azimuth filter
# A: postprocessing filter: source subtraction, azimuth filter
# C and A typically come in groups.
#
# A single logical task may have both A, B and C components, so
# it may be convenient to represent each of these tasks by a filter
# class, with different methods corresponding to these steps.
#
# A problem with joining these is that postprocessing acts on specific
# signals. With the current setup, it is more natural to attach these
# to signals instead ("signal foo needs this postprocessing"). But that
# may be inefficient if multiple postprocessors need to iterate through
# TODs.
#
# The best solution is probably to handle windowing, preprocessing and
# postprocessing individually. Each signal should have an array of
# postprocessors. The equation system has an array of preprocessors.
# And either the eqsys or the nmat itself takes care of windowing.
# Advantages of the latter:
#  1. Windowing can be interpreted as a position-dependence of N
#  2. N is measured assuming a given window, so the window should be
#     tied to it and stored with it.
#  3. It simplifies the mapmaker and other code that needs to use N.
# Disadvantages:
#  1. calibrate must return an unwindowed tod, which means that it
#     either can't use windowing, or it must unwindow first. Unwindowing
#     is only safe if immediately followed by windowing.
#  2. Jacobi preconditioner might want to take windowing into account.
#  3. One might want to use other kinds of weighting, such as az-mode
#     weighting. Moving all of these into the noise matrix would be very
#     messy.

class FilterPickup:
	def __init__(self, daz=None, nt=None, nhwp=None, niter=None):
		self.daz, self.nt, self.nhwp, self.niter = daz, nt, nhwp, niter
	def __call__(self, scan, tod):
		#print "Using weights in FilterPickup. This is slow. Should think about best way to do this."
		#weights = (1-scan.cut_noiseest.to_mask()).astype(tod.dtype)
		if self.daz is None: naz = None
		else:
			waz = (np.max(scan.boresight[:,1])-np.min(scan.boresight[:,1]))/utils.degree
			naz = utils.nint(waz/self.daz)
		todfilter.filter_poly_jon(tod, scan.boresight[:,1], hwp=scan.hwp, naz=naz, nt=self.nt, nhwp=self.nhwp, niter=self.niter, cuts=scan.cut)

class FilterHWPNotch:
	def __init__(self, nmode=1):
		self.nmode     = nmode
		self.harmonics = None
	def __call__(self, scan, tod):
		if not scan.hwp_active: return
		if self.harmonics is None:
			# Determine the HWP rotation frequency
			hwp       = utils.unwind(scan.hwp)
			hwp_freq  = np.abs(hwp[-1]-hwp[0])/(scan.nsamp-1)/(2*np.pi)
			tod_freqs = fft.rfftfreq(scan.nsamp)
			self.harmonics = utils.nint((np.arange(hwp_freq, tod_freqs[-1], hwp_freq)/tod_freqs[1]))
		# Prepare the filter
		ft  = fft.rfft(tod)
		off = self.nmode//2
		for h in self.harmonics:
			ft[...,h-off:h-off+self.nmode] = 0
		return fft.ifft(ft, tod, normalize=True)

class PostPickup:
	def __init__(self, scans, signal_map, signal_cut, prec_ptp, daz=None, nt=None, weighted=False):
		self.scans = scans
		self.signal_map = signal_map
		self.signal_cut = signal_cut
		self.daz, self.nt = daz, nt
		self.ptp = prec_ptp
		self.weighted = weighted
	def __call__(self, imap):
		# This function has a lot of duplicate code with Eqsys.A :/
		signals = [self.signal_cut, self.signal_map]
		imaps   = [self.signal_cut.zeros(), imap]
		omaps   = [signal.zeros() for signal in signals]
		iwork   = [signal.prepare(map) for signal, map in zip(signals, imaps)]
		owork   = [signal.prepare(map) for signal, map in zip(signals, omaps)]
		if self.weighted:
			wmap  = self.ptp.div[0]
			wmap[1:] = 0
			wwork = self.signal_map.prepare(wmap)
		for scan in self.scans:
			tod = np.zeros([scan.ndet, scan.nsamp], self.signal_map.dtype)
			for signal, work in zip(signals, iwork)[::-1]:
				signal.forward(scan, tod, work)
			if self.weighted: 
				# Weighted needs quite a bit more memory :/
				weights = np.zeros([scan.ndet, scan.nsamp], self.signal_map.dtype)
				self.signal_map.forward(scan, weights, wwork)
			else: weights = None
			# I'm worried about the effect of single, high pixels at the edge
			# here. Even when disabling desloping, we may still end up introducing
			# striping when subtracting polynomials fit to data with very
			# inhomogeneous noise. Might it be better to apply the filter to
			# a prewhitened map?
			if self.daz is None: naz = None
			else:
				waz = (np.max(scan.boresight[:,1])-np.min(scan.boresight[:,1]))/utils.degree
				naz = utils.nint(waz/self.daz)
			todfilter.filter_poly_jon(tod, scan.boresight[:,1], weights=weights, naz=naz, nt=self.nt)
			for signal, work in zip(signals, owork):
				signal.backward(scan, tod, work)
			if self.weighted: del weights, tod
		for signal, map, work in zip(signals, omaps, owork):
			signal.finish(map, work)
		# Must use (P'P)" here, not any other preconditioner!
		self.ptp(omaps[1])
		return omaps[1]

class FilterAddMap:
	def __init__(self, scans, map, sys=None, mul=1, tmul=1, pmat_order=None):
		self.map, self.sys, self.mul, self.tmul = map, sys, mul, tmul
		self.data = {scan: pmat.PmatMap(scan, map, order=pmat_order, sys=sys) for scan in scans}
	def __call__(self, scan, tod):
		pmat = self.data[scan]
		pmat.forward(tod, self.map, tmul=self.tmul, mmul=self.mul)

class FilterAddDmap:
	def __init__(self, scans, subinds, dmap, sys=None, mul=1, tmul=1, pmat_order=None):
		self.map, self.sys, self.mul, self.tmul = dmap, sys, mul, tmul
		self.data = {}
		work = dmap.tile2work()
		for scan, subind in zip(scans, subinds):
			self.data[scan] = [pmat.PmatMap(scan, work[subind], order=pmat_order, sys=sys), work[subind]]
	def __call__(self, scan, tod):
		pmat, work = self.data[scan]
		pmat.forward(tod, work, tmul=self.tmul, mmul=self.mul)

class PostAddMap:
	# This one is easy if imap and map are compatible (the common case),
	# but hard otherwise, as it requires reprojection in that case. We
	# only support the compatible case for now.
	def __init__(self, map, mul=1):
		self.map, self.mul = map, mul
	def __call__(self, imap):
		return imap + self.map*self.mul

class FilterBuddy:
	def __init__(self, scans, map, sys=None, mul=1, tmul=1, pmat_order=None):
		self.map, self.sys, self.mul, self.tmul = map, sys, mul, tmul
		self.data = {scan: pmat.PmatMapMultibeam(scan, map, scan.buddy_offs,
			scan.buddy_comps, order=pmat_order, sys=sys) for scan in scans}
	def __call__(self, scan, tod):
		pmat = self.data[scan]
		pmat.forward(tod, self.map, tmul=self.tmul, mmul=self.mul)

class FilterBuddyDmap:
	def __init__(self, scans, subinds, dmap, sys=None, mul=1, tmul=1, pmat_order=None):
		self.map, self.sys, self.mul, self.tmul = dmap, sys, mul, tmul
		self.data = {}
		work = dmap.tile2work()
		for scan, subind in zip(scans, subinds):
			self.data[scan] = [pmat.PmatMapMultibeam(scan, work[subind], scan.buddy_offs,
				scan.buddy_comps, order=pmat_order, sys=sys), work[subind]]
	def __call__(self, scan, tod):
		pmat, work = self.data[scan]
		pmat.forward(tod, work, tmul=self.tmul, mmul=self.mul)

class FilterBuddyPertod:
	def __init__(self, map, sys=None, mul=1, tmul=1, pmat_order=None, nstep=200, prec="bin"):
		self.map, self.sys, self.mul, self.tmul = map, sys, mul, tmul
		self.comm  = mpi.COMM_SELF
		self.dtype = map.dtype
		self.nstep = nstep
		self.pmat_order = pmat_order
		self.prec  = prec
	def __call__(self, scan, tod):
		# We need to be able to restore the noise model, as this is just
		# one among several filters, and the main noise model must be computed
		# afterwards. We currently store the noise model computer as a special
		# kind of noise model that gets overwritten. That's not a good choice,
		# but since that's how we do it, we must preserve this object.
		old_noise  = scan.noise
		# Set up our equation system
		signal_cut = SignalCut([scan], self.dtype, self.comm)
		signal_map = SignalMap([scan], self.map,   self.comm, sys=self.sys, pmat_order=self.pmat_order)
		pmat_buddy =  pmat.PmatMapMultibeam(scan, self.map, scan.buddy_offs,
			scan.buddy_comps, order=self.pmat_order, sys=self.sys)
		eqsys = Eqsys([scan], [signal_cut, signal_map], dtype=self.dtype, comm=self.comm)
		# This also updates the noise model
		eqsys.calc_b(tod.copy())
		# Set up a preconditioner
		if self.prec == "bin" or self.prec == "jacobi":
			signal_map.precon = PreconMapBinned(signal_map, signal_cut, [scan], [], noise=self.prec=="bin", hits=False)
		else: raise NotImplementedError
		# Ok, we can now solve the system
		cg = CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dot)
		for i in range(self.nstep):
			cg.step()
			L.debug("buddy pertod step %3d err %15.7e" % (cg.i, cg.err))
		# Apodize the buddy map
		buddy_map = eqsys.dof.unzip(cg.x)[1]
		div  = signal_map.precon.div[0,0]
		apod = np.minimum(1,div/(0.2*np.median(div[div!=0])))**4
		buddy_map *= apod
		# Ok, the system is hopefully solved by now. Read off the buddy model
		pmat_buddy.forward(tod, buddy_map, tmul=self.tmul, mmul=self.mul)
		# Restore noise
		scan.noise = old_noise
		del signal_map.precon
		del signal_map

class FilterAddSrcs:
	def __init__(self, scans, params, sys=None, mul=1):
		self.params = params
		self.data = {}
		for scan in scans:
			self.data[scan] = pmat.PmatPtsrc(scan, params, sys=sys, pmul=mul)
	def __call__(self, scan, tod):
		pmat = self.data[scan]
		pmat.forward(tod, self.params)

class FilterWindow:
	# Windowing filter tapers the start and end of the TOD
	def __init__(self, width):
		self.width = width
	def __call__(self, scan, tod):
		nsamp = int(self.width*scan.srate)
		nmat.apply_window(tod, nsamp)

class FilterDedark:
	def __init__(self, fit_highpass=0.1):
		self.fit_highpass = fit_highpass
	def __call__(self, scan, tod):
		nmode = int(tod.shape[-1]/2*self.fit_highpass/scan.srate)
		todfilter.deproject_vecs_smooth(tod, scan.dark_tod, nmode=nmode, inplace=True, cuts=scan.cut)

class FilterPhaseBlockwise:
	def __init__(self, daz=None, niter=None):
		self.daz, self.niter = daz, niter
	def __call__(self, scan, tod):
		blocks = utils.find_equal_groups(scan.layout.pcb[scan.dets])
		todfilter.filter_phase_blockwise(tod, blocks, scan.boresight[:,1], daz=self.daz, cuts=scan.cut, niter=self.niter, inplace=True)

class FilterCommonBlockwise:
	def __init__(self, niter=None):
		self.niter = niter
	def __call__(self, scan, tod):
		blocks = utils.find_equal_groups(scan.layout.pcb[scan.dets])
		todfilter.filter_common_blockwise(tod, blocks, cuts=scan.cut, niter=self.niter, inplace=True)

class FilterGapfill:
	def __call__(self, scan, tod):
		gapfill.gapfill(tod, scan.cut, inplace=True)

######## Equation system ########

class Eqsys:
	def __init__(self, scans, signals, filters=[], filters2=[], weights=[], dtype=np.float64, comm=None):
		self.scans   = scans
		self.signals = signals
		self.dtype   = dtype
		self.filters = filters
		self.filters2= filters2
		self.weights = weights
		self.dof     = zipper.MultiZipper([signal.dof for signal in signals], comm=comm)
		self.b       = None
	def A(self, x):
		"""Apply the A-matrix P'N"P to the zipped vector x, returning the result."""
		with bench.mark("A_init"):
			imaps  = self.dof.unzip(x)
			omaps  = [signal.zeros() for signal in self.signals]
			# Set up our input and output work arrays. The output work array will accumulate
			# the results, so it must start at zero.
			iwork = [signal.prepare(map) for signal, map in zip(self.signals, imaps)]
			owork = [signal.work() for signal in self.signals]
			#owork = [signal.prepare(map) for signal, map in zip(self.signals, omaps)]
		for scan in self.scans:
			# Set up a TOD for this scan
			tod = np.zeros([scan.ndet, scan.nsamp], self.dtype)
			# Project each signal onto the TOD (P) in reverse order. This is done
			# so that the cuts can override the other signals.
			with bench.mark("A_P"):
				for signal, work in zip(self.signals, iwork)[::-1]:
					with bench.mark("A_Pr_" + signal.name):
						signal.precompute(scan)
					with bench.mark("A_P_" + signal.name):
						signal.forward(scan, tod, work)
			# Apply the noise matrix (N")
			with bench.mark("A_N"):
				for weight in self.weights: weight(scan, tod)
				scan.noise.apply(tod)
				for weight in self.weights[::-1]: weight(scan, tod)
			# Project the TOD onto each signal (P') in normal order. This is done
			# to allow the cuts to zero out the relevant TOD samples first
			with bench.mark("A_PT"):
				for signal, work in zip(self.signals, owork):
					with bench.mark("A_PT_" + signal.name):
						signal.backward(scan, tod, work)
					with bench.mark("A_Fr_" + signal.name):
						signal.free()
			times = [bench.stats[s]["time"].last for s in ["A_P","A_N","A_PT"]]
			L.debug("A P %5.3f N %5.3f P' %5.3f %s %4d" % (tuple(times)+(scan.id,scan.ndet)))
		# Collect all the results, and flatten them
		with bench.mark("A_reduce"):
			for signal, map, work in zip(self.signals, omaps, owork):
				signal.finish(map, work)
		# priors
		with bench.mark("A_prior"):
			for signal, imap, omap in zip(self.signals, imaps, omaps):
				signal.prior(self.scans, imap, omap)
		return self.dof.zip(omaps)
	def M(self, x):
		"""Apply the preconditioner to the zipped vector x."""
		with bench.mark("M"):
			maps = self.dof.unzip(x)
			for signal, map in zip(self.signals, maps):
				signal.precon(map)
			return self.dof.zip(maps)
	def calc_b(self, itod=None):
		"""Compute b = P'N"d, and store it as the .b member. This involves
		reading in the TOD data and potentially estimating a noise model,
		so it is a heavy operation."""
		maps  = [signal.zeros() for signal in self.signals]
		owork = [signal.work() for signal in self.signals]
		#owork = [signal.prepare(map) for signal, map in zip(self.signals,maps)]
		for scan in self.scans:
			# Get the actual TOD samples (d)
			if itod is None:
				with bench.mark("b_read"):
					tod  = scan.get_samples()
					#tod -= np.copy(tod[:,0,None])
					tod  = tod.astype(self.dtype)
			else: tod = itod
			#dump("dump_prefilter.hdf", tod[:4])
			# Apply all filters (pickup filter, src subtraction, etc)
			with bench.mark("b_filter"):
				for filter in self.filters: filter(scan, tod)
			#dump("dump_postfilter.hdf", tod[:4])
			#1/0
			# Apply the noise model (N")
			with bench.mark("b_weight"):
				for weight in self.weights: weight(scan, tod)
			#dump("dump_prenoise.hdf", tod[:32])
			with bench.mark("b_N_build"):
				scan.noise = scan.noise.update(tod, scan.srate)
				#print "FIXME"
				#sampcut.gapfill_const(scan.cut, tod, 0.0, True)
			with bench.mark("b_filter2"):
				for filter in self.filters2: filter(scan, tod)
			with bench.mark("b_N"):
				scan.noise.apply(tod)
			with bench.mark("b_weight"):
				for weight in self.weights[::-1]: weight(scan, tod)
			# Project onto signals
			with bench.mark("b_PT"):
				for signal, work in zip(self.signals, owork):
					with bench.mark("b_PT_" + signal.name):
						signal.precompute(scan)
						signal.backward(scan, tod, work)
						signal.free()
			del tod
			times = [bench.stats[s]["time"].last for s in ["b_read","b_filter","b_N_build", "b_N", "b_PT"]]
			L.debug("b get %5.1f f %5.1f NB %5.3f N %5.3f P' %5.3f %s" % (tuple(times)+(scan.id,)))
		# Collect results
		with bench.mark("b_reduce"):
			for signal, map, work in zip(self.signals, maps, owork):
				signal.finish(map, work)
		with bench.mark("b_zip"):
			self.b = self.dof.zip(maps)
	def dot(self, a, b):
		with bench.mark("dot"):
			return self.dof.dot(a,b)
	def postprocess(self, x):
		maps = self.dof.unzip(x)
		for i in range(len(self.signals)):
			maps[i] = self.signals[i].postprocess(maps[i])
		return self.dof.zip(maps)
	def write(self, prefix, tag, x):
		maps = self.dof.unzip(x)
		for signal, map in zip(self.signals, maps):
			signal.write(prefix, tag, map)
	# These debug functions don't work properly for
	# distributed degrees of freedom.
	def check_symmetry(self, inds):
		"""Debug function - checks the symmetry of A[inds,inds]"""
		res = np.zeros([len(inds),len(inds)])
		for i, ind in enumerate(inds):
			ivec = np.zeros(self.dof.n)
			ivec[ind] = 1
			ovec = self.A(ivec)
			res[i] = ovec[inds]
			if self.dof.comm.rank == 0:
				print "----", np.sum(ovec), ind
				np.savetxt("/dev/stdout", res[:i+1,:i+1], fmt="%11.4e")
	def calc_A(self):
		n = self.dof.n
		res = np.eye(n)
		for i in range(n):
			res[i] = self.A(res[i])
		return res
	def calc_M(self):
		n = self.dof.n
		res = np.eye(n)
		for i in range(n):
			res[i] = self.M(res[i])
		return res

def write_precons(signals, prefix):
	for signal in signals:
		signal.precon.write(prefix)
