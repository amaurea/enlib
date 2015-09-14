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

######## Signals ########

class Signal:
	"""General signal interface."""
	def __init__(self, scans):
		self.dof    = zipper.ArrayZipper(np.zeros([0]))
		self.precon = PreconNull()
	def prepare (self, x): return x.copy()
	def forward (self, scan, tod, x): pass
	def backward(self, scan, tod, x): pass
	def finish  (self, x, y): x[:] = y

def SignalMap(Signal):
	def __init__(self, scans, area, cuts=None, comm=None, pmat_order=None, eqsys=None):
		self.area = area
		self.cuts = cuts
		self.dof  = dmap.ArrayZipper(area, shared=False)
		self.data = {scan: pmat.PmatMap(scan, area, order=pmat_order, eqsys=eqsys) for scan in scans}
	def forward(self, scan, tod, work):
		self.data[scan].forward(tod, work)
	def backward(self, scan, tod, work):
		self.data[scan].backward(tod, work)
	def finish(self, m, work):
		self.dof.comm.Allreduce(work, m)
	def zeros(self): return enmap.zeros(self.area.shape, self.area.wcs, self.area.dtype)

def SignalDMap2(Signal):
	def __init__(self, scans, subinds, area, cuts=None, comm=None, pmat_order=None, eqsys=None):
		self.area = area
		self.cuts = cuts
		self.dof  = dmap.DmapZipper(area)
		self.subs = subinds
		self.data = {}
		for scan, subind in zip(scans, subinds):
			work = area.tile2work()
			self.data[scan] = [pmat.PmatMap(scan, work[subind], order=pmat_order, eqsys=eqsys), subind]
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
	def zeros(self): return dmap.Dmap2(self.area.geometry)

def SignalCut(Signal):
	def __init__(self, scans, dtype, cut_type=None):
		self.data  = {}
		self.dtype = dtype
		cutrange = [0,0]
		for scan in scans:
			mat = pmat.PmatCut(scan, cut_type)
			cutrange = [cutrange[1], cutrange[1]+mat.njunk]
			self.data[scan] = [mat, cutrange]
		self.njunk = cutrange[1]
	def forward(self, scan, tod, junk):
		mat, cutrange = self.data[scan]
		mat.forward(tod, junk[cutrange[0]:cutrange[1]])
	def backward(self, scan, tod, junk):
		mat, cutrange = self.data[scan]
		mat.backward(tod, junk[cutrange[0]:cutrange[1]])
	def zeros(self): return np.zeros(self.njunk, self.dtype)

######## Preconditioners ########

class PreconNull:
	def __init__(self): pass
	def __call__(self, m): pass
	def write(self, prefix="", ext="fits"): pass

class PreconMapBinned:
	def __init__(self, signal, scans):
		ncomp = signal.area.shape[0]
		div  = enmap.zeros((ncomp,)+signal.area.shape, signal.area.wcs, signal.area.dtype)
		# The cut samples are included here becuase they must be avoided, but the
		# actual computation of the junk sample preconditioner happens elsewhere.
		# This is a bit redundant, but should not cost much time since this only happens
		# in the beginning.
		ijunk= np.zeros(signal.cuts.njunk, dtype=mapeq.area.dtype)
		ojunk= signal.cuts.prepare(signal.cuts.zeros())
		for i in range(ncomp):
			div[i,i] = 1
			iwork = signal.prepare(div[i])
			owork = signal.zeros()
			for scan in scans:
				tod = np.zeros((scan.ndet, scan.nsamp), iwork.dtype)
				signal.forward(scan, tod, iwork)
				signal.cuts.forward (scan, tod, ijunk)
				scan.noise.white(tod)
				signal.cuts.backward(scan, tod. ojunk)
				signal.backward(scan, tod, owork)
			signal.finish(div[i], owork)
		idiv = array_ops.eigpow(div, -1, axes=(0,1))
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
	def __call__(self, m):
		m[:] = array_ops.matmul(self.idiv, m)
	def write(self, prefix="", ext="fits"):
		# We want the original div, not the inverse one
		enmap.write_map(prefix + "div."  + ext, self.div)
		enmap.write_map(prefix + "hits." + ext, self.hits)

class PreconDmap2Binned:
	def __init__(self, signal, scans):
		ncomp = signal.area.shape[0]
		geom  = signal.area.geometry.copy()
		geom.pre = (ncomp,)+geom.pre
		div   = dmap.Dmap2(geom)
		# The cut samples are included here becuase they must be avoided, but the
		# actual computation of the junk sample preconditioner happens elsewhere.
		# This is a bit redundant, but should not cost much time since this only happens
		# in the beginning.
		ijunk= signal.cuts.zeros()
		ojunk= signal.cuts.zeros()
		for i in range(ncomp):
			div[i,i] = 1
			iwork = signal.prepare(div[i])
			owork = signal.prepare(div[i])
			for scan in scans:
				tod = np.zeros((scan.ndet, scan.nsamp), iwork.dtype)
				signal.forward(scan, tod, iwork)
				signal.cuts.forward (scan, tod, ijunk)
				scan.noise.white(tod)
				signal.cuts.backward(scan, tod. ojunk)
				signal.backward(scan, tod, owork)
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
	def __call__(self, m):
		for idtile, mtile in zip(self.idiv.tiles, m.tiles):
			mtile[:] = array_ops.matmul(idtile, mtile, axes=[0,1])
	def write(self, prefix="", ext="fits"):
		# We want the original div, not the inverse one
		dmap.write_map(prefix + "div."  + ext, self.div)
		dmap.write_map(prefix + "hits." + ext, self.hits)

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
	def __call__(self, m):
		m *= self.idiv
	def write(self, prefix="", ext="fits"): pass

######## Equation system ########

class Eqsys:
	def __init__(self, scans, signals, dtype, comm=None):
		self.scans   = scans
		self.signals = signals
		self.dtype   = dtype
		self.dof     = zipper.MultiZipper([signal.dof for signal in signals], comm=comm)
	def A(self, x):
		"""Apply the A-matrix P'N"P to the zipped vector x, returning the result."""
		maps  = self.dof.unzip(x)
		# Set up our input and output work arrays. The output work array will accumulate
		# the results, so it must start at zero.
		iwork = [signal.prepare(map) for signal, map in zip(self.signals, maps)]
		owork = [signal.prepare(signal.zeros()) for signal in signals]
		for scan in self.scans:
			# Set up a TOD for this scan
			tod = np.zeros([scan.ndet, scan.nsamp], self.dtype)
			# Project each signal onto the TOD (P) in reverse order. This is done
			# so that the cuts can override the other signals.
			for signal, work in zip(signals, iwork):
				signal.forward(scan, tod, work)
			# Apply the noise matrix (N")
			scan.noise.apply(tod)
			# Project the TOD onto each signal (P') in normal order. This is done
			# to allow the cuts to zero out the relevant TOD samples first
			for signal, work in zip(singals, owork):
				signal.backward(scan, tod, owork)
		# Collect all the results, and flatten them
		for signal, map, work in zip(signals, maps, owork):
			signal.finish(map, work)
		# NOTE Any priors should be handled here!
		return self.dof.zip(*maps)
	def M(self, x):
		"""Apply the preconditioner to the zipped vector x."""
		maps = self.dof.unzip(x)
		for signal, map in zip(self.signals, maps):
			signal.precon(map)
		return self.dof.zip(maps)
	def calc_b(self):
		maps = [signal.zeros() for signal in self.signals]
		for scan in self.scans:
			# Get the actual TOD samples (d)
			tod  = scan.get_samples()
			tod -= np.mean(tod,1)[:,None]
			tod  = tod.astype(self.dtype)
			# Apply the noise model (N")
			scan.noise.apply(tod)
			# Project onto signals
			for signal, work in zip(self.signals, owork):
				signal.backward(scan, tod, owork)
			del tod
		# Collect results
		for signal, map, work in zip(self.signals, maps, owork):
			signal.finish(map, work)
		return self.dof.zip(maps)
