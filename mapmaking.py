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
import numpy as np, h5py, zipper
from enlib import enmap, dmap2 as dmap, array_ops, pmat, utils

######## Signals ########

class Signal:
	"""General signal interface."""
	def __init__(self, name, ext):
		self.ext    = ext
		self.dof    = zipper.ArrayZipper(np.zeros([0]))
		self.precon = PreconNull()
		self.prior  = PriorNull()
		if isinstance(name, basestring):
			self.name, self.oname = name, name
		else:
			self.name, self.oname = name
	def prepare (self, x): return x.copy()
	def forward (self, scan, tod, x): pass
	def backward(self, scan, tod, x): pass
	def finish  (self, x, y): x[:] = y
	def write   (self, prefix, tag, x): pass

class SignalMap(Signal):
	def __init__(self, scans, area, comm, cuts=None, name="main", ext="fits", pmat_order=None, eqsys=None, nuisance=False):
		Signal.__init__(self, name, ext)
		self.area = area
		self.cuts = cuts
		self.dof  = zipper.ArrayZipper(area, comm=comm)
		self.dtype= area.dtype
		self.data = {scan: pmat.PmatMap(scan, area, order=pmat_order, sys=eqsys) for scan in scans}
	def forward(self, scan, tod, work):
		self.data[scan].forward(tod, work)
	def backward(self, scan, tod, work):
		self.data[scan].backward(tod, work)
	def finish(self, m, work):
		self.dof.comm.Allreduce(work, m)
	def zeros(self): return enmap.zeros(self.area.shape, self.area.wcs, self.area.dtype)
	def write(self, prefix, tag, m):
		if self.oname is None: return
		oname = "%s%s_%s.%s" % (prefix, self.oname, tag, self.ext)
		if self.dof.comm.rank == 0:
			enmap.write_map(oname, m)

class SignalDmap(Signal):
	def __init__(self, scans, subinds, area, cuts=None, name="main", ext="fits", pmat_order=None, eqsys=None, nuisance=False):
		Signal.__init__(self, name, ext)
		self.area = area
		self.cuts = cuts
		self.dof  = dmap.DmapZipper(area)
		self.dtype= area.dtype
		self.subs = subinds
		self.data = {}
		for scan, subind in zip(scans, subinds):
			work = area.tile2work()
			self.data[scan] = [pmat.PmatMap(scan, work[subind], order=pmat_order, sys=eqsys), subind]
	def prepare(self, m):
		return m.tile2work()
	def forward(self, scan, tod, work):
		mat, ind = self.data[scan]
		mat.forward(tod, work[ind])
	def backward(self, scan, tod, work):
		mat, ind = self.data[scan]
		mat.backward(tod, work[ind])
	def finish(self, m, work):
		m.work2tile(work)
	def zeros(self): return dmap.zeros(self.area.geometry)
	def write(self, prefix, tag, m):
		if self.oname is None: return
		oname = "%s%s_%s.%s" % (prefix, self.oname, tag, self.ext)
		dmap.write_map(oname, m)

class SignalCut(Signal):
	def __init__(self, scans, dtype, comm, name=["cut",None], cut_type=None):
		Signal.__init__(self, name, "hdf")
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
		mat, cutrange = self.data[scan]
		mat.forward(tod, junk[cutrange[0]:cutrange[1]])
	def backward(self, scan, tod, junk):
		mat, cutrange = self.data[scan]
		mat.backward(tod, junk[cutrange[0]:cutrange[1]])
	def zeros(self): return np.zeros(self.njunk, self.dtype)
	def write(self, prefix, tag, m):
		if self.oname is None: return
		oname = "%s%s_%s.%s" % (prefix, self.oname.format(rank=self.dof.comm.rank), tag, self.ext)
		with h5py.File(oname, "w") as hfile:
			hfile["data"] = m

class SignalPhase(Signal):
	def __init__(self, scans, pids, patterns, array_shape, res, dtype, comm, cuts=None,
			name=["phase","phase_{pid:02}_{az0:.0}_{az1:.0}_{el:.0}"], col_major=True, hysteresis=True):
		Signal.__init__(self, name, "fits")
		nrow,ncol = array_shape
		ndet = nrow*ncol
		self.pids = pids
		self.patterns = patterns
		self.comm = comm
		self.dtype = dtype
		self.array_shape = array_shape
		self.col_major = col_major
		self.cuts = cuts
		self.data = {}
		self.areas = []
		# Set up an area for each scanning pattern. We assume that these are constant
		# elevation scans, so only azimuth matters. This setup is ugly and would be
		# nicer if passed in, but then the ugliness would only be moved to the calling
		# code instead.
		for pattern in patterns:
			az0,az1 = pattern[:,1]
			naz = np.ceil((az1-az0)/res)
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
		pid, mat = self.data[scan]
		mat.forward(tod, work[pid])
	def backward(self, scan, tod, work):
		pid, mat = self.data[scan]
		mat.backward(tod, work[pid])
	def finish(self, ms, work):
		for m, w in zip(ms, work):
			self.dof.comm.Allreduce(w,m)
	def zeros(self):
		return [area*0 for area in self.areas]
	def write(self, prefix, tag, ms):
		if self.oname is None: return
		fmt = "%s%s_%s.%s" % (prefix, self.oname, tag, self.ext)
		for pid, (pattern, m) in enumerate(zip(self.patterns, ms)):
			oname = fmt.format(pid=pid, el=pattern[0,0]/utils.degree, az0=pattern[0,1]/utils.degree, az1=pattern[1,1]/utils.degree)
			if self.dof.comm.rank == 0:
				enmap.write_map(oname, m)

######## Preconditioners ########

class PreconNull:
	def __init__(self): pass
	def __call__(self, m): pass
	def write(self, prefix="", ext="fits"): pass

def prec_div_helper(signal, scans, iwork, owork, ijunk, ojunk):
	for scan in scans:
		tod = np.zeros((scan.ndet, scan.nsamp), signal.dtype)
		signal.forward(scan, tod, iwork)
		signal.cuts.forward (scan, tod, ijunk)
		scan.noise.white(tod)
		signal.cuts.backward(scan, tod, ojunk)
		signal.backward(scan, tod, owork)

class PreconMapBinned:
	def __init__(self, signal, scans):
		ncomp = signal.area.shape[0]
		div  = enmap.zeros((ncomp,)+signal.area.shape, signal.area.wcs, signal.area.dtype)
		# The cut samples are included here becuase they must be avoided, but the
		# actual computation of the junk sample preconditioner happens elsewhere.
		# This is a bit redundant, but should not cost much time since this only happens
		# in the beginning.
		ijunk= np.zeros(signal.cuts.njunk, dtype=signal.area.dtype)
		ojunk= signal.cuts.prepare(signal.cuts.zeros())
		for i in range(ncomp):
			div[i,i] = 1
			iwork = signal.prepare(div[i])
			owork = signal.prepare(signal.zeros())
			prec_div_helper(signal, scans, iwork, owork, ijunk, ojunk)
			signal.finish(div[i], owork)
		idiv = array_ops.eigpow(div, -1, axes=[0,1])
		# Build hitcount map too
		hits = signal.area.copy()
		work = signal.prepare(hits)
		for scan in scans:
			tod = np.full((scan.ndet, scan.nsamp), 1, hits.dtype)
			signal.cuts.backward(scan, tod, ojunk)
			signal.backward(scan, tod, work)
		signal.finish(hits, work)
		hits = hits[0].astype(np.int32)
		self.div, self.idiv, self.hits = div, idiv, hits
		self.signal = signal
	def __call__(self, m):
		m[:] = array_ops.matmul(self.idiv, m, axes=[0,1])
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)
		self.signal.write(prefix, "hits", self.hits)

class PreconDmapBinned:
	def __init__(self, signal, scans):
		ncomp = signal.area.shape[0]
		geom  = signal.area.geometry.copy()
		geom.pre = (ncomp,)+geom.pre
		div   = dmap.zeros(geom)
		# The cut samples are included here becuase they must be avoided, but the
		# actual computation of the junk sample preconditioner happens elsewhere.
		# This is a bit redundant, but should not cost much time since this only happens
		# in the beginning.
		ijunk= signal.cuts.zeros()
		ojunk= signal.cuts.zeros()
		for i in range(ncomp):
			div[i,i] = 1
			iwork = signal.prepare(div[i])
			owork = signal.prepare(signal.zeros())
			prec_div_helper(signal, scans, iwork, owork, ijunk, ojunk)
			signal.finish(div[i], owork)
		idiv = div.copy()
		for dtile in idiv.tiles:
			dtile[:] = array_ops.eigpow(dtile, -1, axes=[0,1])
		# Build hitcount map too
		hits = signal.area.copy()
		work = signal.prepare(hits)
		for scan in scans:
			tod = np.full((scan.ndet, scan.nsamp), 1, hits.dtype)
			signal.cuts.backward(scan, tod, ojunk)
			signal.backward(scan, tod, work)
		signal.finish(hits, work)
		hits = hits[0].astype(np.int32)
		self.div, self.idiv, self.hits = div, idiv, hits
		self.signal = signal
	def __call__(self, m):
		for idtile, mtile in zip(self.idiv.tiles, m.tiles):
			mtile[:] = array_ops.matmul(idtile, mtile, axes=[0,1])
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)
		self.signal.write(prefix, "hits", self.hits)

class PreconCut:
	def __init__(self, signal, scans):
		junk  = signal.zeros()
		iwork = signal.prepare(junk)
		owork = signal.prepare(junk)
		for scan in scans:
			tod = np.zeros((scan.ndet, scan.nsamp), iwork.dtype)
			signal.forward (scan, tod, iwork)
			scan.noise.white(tod)
			signal.backward(scan, tod, owork)
		signal.finish(junk, owork)
		self.idiv = 1/junk
		self.signal = signal
	def __call__(self, m):
		m *= self.idiv
	def write(self, prefix):
		self.signal.write(prefix, "idiv", self.idiv)

class PreconPhaseBinned:
	def __init__(self, signal, scans):
		div = [area*0+1 for area in signal.areas]
		# The cut samples are included here becuase they must be avoided, but the
		# actual computation of the junk sample preconditioner happens elsewhere.
		# This is a bit redundant, but should not cost much time since this only happens
		# in the beginning.
		ijunk = np.zeros(signal.cuts.njunk, dtype=signal.dtype)
		ojunk = signal.cuts.prepare(signal.cuts.zeros())
		iwork = signal.prepare(div)
		owork = signal.prepare(signal.zeros())
		prec_div_helper(signal, scans, iwork, owork, ijunk, ojunk)
		signal.finish(div, owork)
		self.div = div
		self.signal = signal
	def __call__(self, ms):
		for d, m in zip(self.div, ms):
			m[d!=0] /= d[d!=0]
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)

######## Priors ########

class PriorNull:
	def __call__(self, xin, xout): pass

class PriorMapNohor:
	def __init__(self, weight=1):
		self.weight = weight
	def __call__(self, imap, omap):
		omap += np.asarray(np.sum(imap*self.weight,-1))[:,:,None]*self.weight

class PriorDmapNohor:
	def __init__(self, weight=1):
		self.weight = weight
	def __call__(self, imap, omap):
		tmp = omap*0
		dmap.broadcast_into(tmp, dmap.sum(imap*self.weight,-1), -1)
		omap += tmp * self.weight

######## Equation system ########

class Eqsys:
	def __init__(self, scans, signals, dtype, comm=None):
		self.scans   = scans
		self.signals = signals
		self.dtype   = dtype
		self.dof     = zipper.MultiZipper([signal.dof for signal in signals], comm=comm)
		self.b       = None
	def A(self, x):
		"""Apply the A-matrix P'N"P to the zipped vector x, returning the result."""
		imaps  = self.dof.unzip(x)
		omaps  = [signal.zeros() for signal in self.signals]
		# Set up our input and output work arrays. The output work array will accumulate
		# the results, so it must start at zero.
		iwork = [signal.prepare(map) for signal, map in zip(self.signals, imaps)]
		owork = [signal.prepare(map) for signal, map in zip(self.signals, omaps)]
		for scan in self.scans:
			# Set up a TOD for this scan
			tod = np.zeros([scan.ndet, scan.nsamp], self.dtype)
			# Project each signal onto the TOD (P) in reverse order. This is done
			# so that the cuts can override the other signals.
			for signal, work in zip(self.signals, iwork):
				signal.forward(scan, tod, work)
			# Apply the noise matrix (N")
			scan.noise.apply(tod)
			# Project the TOD onto each signal (P') in normal order. This is done
			# to allow the cuts to zero out the relevant TOD samples first
			for signal, work in zip(self.signals, owork):
				signal.backward(scan, tod, work)
		# Collect all the results, and flatten them
		for signal, map, work in zip(self.signals, omaps, owork):
			signal.finish(map, work)
		# priors
		for signal, imap, omap in zip(self.signals, imaps, omaps):
			signal.prior(imap, omap)
		return self.dof.zip(omaps)
	def M(self, x):
		"""Apply the preconditioner to the zipped vector x."""
		maps = self.dof.unzip(x)
		for signal, map in zip(self.signals, maps):
			signal.precon(map)
		return self.dof.zip(maps)
	def calc_b(self):
		maps  = [signal.zeros() for signal in self.signals]
		owork = [signal.prepare(map) for signal, map in zip(self.signals,maps)]
		for scan in self.scans:
			# Get the actual TOD samples (d)
			tod  = scan.get_samples()
			tod -= np.mean(tod,1)[:,None]
			tod  = tod.astype(self.dtype)
			# Apply the noise model (N")
			scan.noise.apply(tod)
			# Project onto signals
			for signal, work in zip(self.signals, owork):
				signal.backward(scan, tod, work)
			del tod
		# Collect results
		for signal, map, work in zip(self.signals, maps, owork):
			signal.finish(map, work)
		self.b = self.dof.zip(maps)
	def write(self, prefix, tag, x):
		maps = self.dof.unzip(x)
		for signal, map in zip(self.signals, maps):
			signal.write(prefix, tag, map)

def write_precons(signals, prefix):
	for signal in signals:
		signal.precon.write(prefix)
