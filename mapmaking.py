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
from __future__ import division, print_function
import numpy as np, h5py, logging, gc
from . import enmap, dmap, array_ops, pmat, utils, todfilter, pointsrcs, zipper
from . import config, nmat, bench, gapfill, mpi, sampcut, fft, bunch
from .cg import CG
L = logging.getLogger(__name__)

def dump(fname, d):
	print("dumping " + fname)
	with h5py.File(fname, "w") as hfile:
		hfile["data"] = d

config.default("debug_raw",False,"moo")

######## Signals ########

class Signal:
	"""General signal interface."""
	def __init__(self, name, ofmt, output, ext):
		self.ext    = ext
		self.dof    = zipper.ArrayZipper(np.zeros([0]))
		self.precon = PreconNull()
		self.prior  = PriorNull()
		self.post   = []
		self.filters= []
		self.name   = name
		self.ofmt   = ofmt
		self.output = output
	def prepare (self, x): return x.copy()
	def forward (self, scan, tod, x): pass
	def backward(self, scan, tod, x): pass
	def precompute(self, scan): pass
	def free(self): pass
	def finish  (self, x, y): x[:] = y
	def polmat(self): return self.zeros(mat=True)
	def transform(self, m, op): return op(m)
	def polinv(self, m): return 1/m
	def polmul(self, mat, m, inplace=False): return mat*m
	# This one is a potentially cheaper version of self.prepare(self.zeros())
	def work    (self, mat=False): return self.prepare(self.zeros(mat=mat))
	def write   (self, prefix, tag, x): pass
	def postprocess(self, x):
		for p in self.post: x = p(x)
		return x
	def filter(self, x):
		for f in self.filters:
			x = f(x)
		return x

class SignalMap(Signal):
	def __init__(self, scans, area, comm, cuts=None, name="main", ofmt="{name}", output=True,
			ext="fits", pmat_order=None, sys=None, nuisance=False, data=None, extra=[], split=False):
		Signal.__init__(self, name, ofmt, output, ext)
		self.area = area
		self.cuts = cuts
		self.dof  = zipper.ArrayZipper(area, comm=comm)
		self.dtype= area.dtype
		self.split= split
		if data is not None:
			self.data = data
		else:
			self.data = {scan: pmat.PmatMap(scan, area, order=pmat_order, sys=sys, extra=extra, split=scan.split if split else None) for scan in scans}
	def forward(self, scan, tod, work, tmul=1, mmul=1):
		if scan not in self.data: return
		self.data[scan].forward(tod, work, tmul=tmul, mmul=mmul)
	def backward(self, scan, tod, work, tmul=1, mmul=1):
		if scan not in self.data: return
		self.data[scan].backward(tod, work, tmul=tmul, mmul=mmul)
	def finish(self, m, work):
		if m.flags["C_CONTIGUOUS"]:
			self.dof.comm.Allreduce(work, m)
		else:
			tmp = np.ascontiguousarray(m)
			self.dof.comm.Allreduce(work, tmp)
			m[:] = tmp
	def zeros(self, mat=False):
		shape = self.area.shape
		if mat: shape = shape[:-2]+shape[-3:]
		return enmap.zeros(shape, self.area.wcs, self.area.dtype)
	def polinv(self, m): return array_ops.eigpow(m, -1, axes=[-4,-3], lim=config.get("eig_limit"), fallback="scalar")
	def polmul(self, mat, m, inplace=False):
		if not inplace: m = m.copy()
		m[:] = array_ops.matmul(mat, m, axes=[-4,-3])
		return m
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		if self.dof.comm.rank == 0:
			enmap.write_map(oname, m)

class SignalMapFast(SignalMap):
	def __init__(self, scans, area, comm, cuts=None, name="main", ofmt="{name}", output=True,
			ext="fits", sys=None, nuisance=False, data=None, extra=[]):
		if data is None:
			data = {scan: pmat.PmatMapFast(scan, area, sys=sys, extra=extra) for scan in scans}
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
			ext="fits", pmat_order=None, sys=None, nuisance=False, data=None, extra=[]):
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
				data[scan] = [pmat.PmatMap(scan, work.maps[subind], order=pmat_order, sys=sys, extra=extra), subind]
		self.data = data
	def prepare(self, m): return m.tile2work()
	def forward(self, scan, tod, work, tmul=1, mmul=1):
		if scan not in self.data: return
		mat, ind = self.data[scan]
		mat.forward(tod, work.maps[ind], tmul=tmul, mmul=mmul)
	def backward(self, scan, tod, work, tmul=1, mmul=1):
		if scan not in self.data: return
		mat, ind = self.data[scan]
		mat.backward(tod, work.maps[ind], tmul=tmul, mmul=mmul)
	def finish(self, m, work):
		m.work2tile(work)
	def filter(self, work):
		res = []
		for w in work.maps:
			for filter in self.filters:
				w = filter(w)
			res.append(w)
		return dmap.Workspace(res)
	def zeros(self, mat=False):
		geom = self.area.geometry
		if mat:
			geom = geom.copy()
			geom.pre = geom.pre + geom.pre[-1:]
		return dmap.zeros(geom)
	def transform(self, imap, op):
		omap = imap.copy()
		for dtile in omap.tiles:
			dtile[:] = op(dtile)
		return omap
	def polinv(self, idiv):
		odiv = idiv.copy()
		for dtile in odiv.tiles:
			dtile[:] = array_ops.eigpow(dtile, -1, axes=[-4,-3], lim=config.get("eig_limit"), fallback="scalar")
		return odiv
	def polmul(self, mat, m, inplace=False):
		if not inplace: m = m.copy()
		for idtile, mtile in zip(mat.tiles, m.tiles):
			mtile[:] = array_ops.matmul(idtile, mtile, axes=[-4,-3])
		return m
	def work(self):  return self.area.geometry.build_work()
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		merged= config.get("dmap_format") == "merged"
		dmap.write_map(oname, m, merged=merged)

class SignalDmapFast(SignalDmap):
	def __init__(self, scans, subinds, area, cuts=None, name="main", ofmt="{name}", output=True,
			ext="fits", sys=None, nuisance=False, data=None, extra=[]):
		if data is None:
			data = {}
			work = area.tile2work()
			for scan, subind in zip(scans, subinds):
				data[scan] = [pmat.PmatMapFast(scan, work.maps[subind], sys=sys, extra=extra), subind]
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
		mat.forward(tod, work.maps[ind], self.pix, self.phase)
	def backward(self, scan, tod, work):
		if scan not in self.data: return
		mat, ind = self.data[scan]
		mat.backward(tod, work.maps[ind], self.pix, self.phase)

class SignalCut(Signal):
	def __init__(self, scans, dtype, comm, name="cut", ofmt="{name}_{rank:02}", output=False, cut_type=None, cuts=None, keep=False):
		Signal.__init__(self, name, ofmt, output, ext="hdf")
		self.data  = {}
		self.dtype = dtype
		cutrange = [0,0]
		for si, scan in enumerate(scans):
			# Use cuts from scan by default, but allow override
			cut = None if cuts is None else cuts[si]
			mat = pmat.PmatCut(scan, cut_type, tmul=keep*1.0, cut=cut)
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
	def zeros(self, mat=False): return np.zeros(self.njunk, self.dtype)
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name, rank=self.dof.comm.rank)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		with h5py.File(oname, "w") as hfile:
			hfile["data"] = m

class SignalNoiseRect(Signal):
	def __init__(self, scans, area, yspeed, yoffs, comm, cuts=None, mode="keepaz", name="noiserect", ofmt="{name}", output=True,
			ext="fits", pmat_order=None, sys=None, nuisance=False, data=None, extra=[]):
		Signal.__init__(self, name, ofmt, output, ext)
		self.area = area
		self.cuts = cuts
		self.yspeed = yspeed
		self.yoffs  = yoffs
		self.mode   = mode
		self.dof  = zipper.ArrayZipper(area, comm=comm)
		self.dtype= area.dtype
		if data is not None:
			self.data = data
		else:
			self.data = {}
			for i, scan in enumerate(scans):
				self.data[scan] = pmat.PmatNoiseRect(scan, area, yspeed, yoffs[i], mode=mode, extra=extra)
	def forward(self, scan, tod, work):
		if scan not in self.data: return
		self.data[scan].forward(tod, work)
	def backward(self, scan, tod, work):
		if scan not in self.data: return
		self.data[scan].backward(tod, work)
	def finish(self, m, work):
		self.dof.comm.Allreduce(work, m)
	def zeros(self, mat=False):
		shape = self.area.shape
		if mat: shape = shape[-2:] + shape
		return enmap.zeros(shape, self.area.wcs, self.area.dtype)
	def polinv(self, m): return array_ops.eigpow(m, -1, axes=[0,1], lim=config.get("eig_limit"), fallback="scalar")
	def polmul(self, mat, m, inplace=False):
		if not inplace: m = m.copy()
		m[:] = array_ops.matmul(mat, m, axes=[0,1])
		return m
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		if self.dof.comm.rank == 0:
			enmap.write_map(oname, m)

class PhaseMap:
	"""Helper class for SignalPhase. Represents a set of [...,ndet,naz] phase "maps",
	one per scanning pattern."""
	def __init__(self, patterns, dets, maps):
		self.patterns = patterns
		self.dets    = dets
		self.maps    = maps
	@staticmethod
	def read(dirname, rewind=False):
		dets     = np.loadtxt(dirname + "/dets.txt").astype(int)
		patterns = []
		maps     = []
		with open(dirname + "/info.txt", "r") as f:
			for line in f:
				line = line.strip()
				if len(line) == 0 or line.startswith("#"): continue
				toks = line.split()
				el, az1, az2 = [float(w)*utils.degree for w in toks[:3]]
				map  = enmap.read_map(dirname + "/" + toks[3])
				if rewind:
					off  = utils.rewind(az1)-az1
					az1, az2 = az1+off, az2+off
					maz  = map.pix2sky([0,0])[1]
					off2 = utils.rewind(maz)-maz
					map.wcs.wcs.crval[0] += off2/utils.degree
					maz2 = map.pix2sky([0,0])[1]
					print("pattern rewind off1 %8.3f off2 %8.3f diff %8.3f maz %8.3f maz2 %8.3f" % (off/utils.degree, off2/utils.degree, (off-off2)/utils.degree, maz/utils.degree, maz2/utils.degree))
				patterns.append([[el,az1],[el,az2]])
				maps.append(map)
		return PhaseMap(patterns, dets, maps)
	def write(self, dirname, fmt="{pid:02}_{az0:.0f}_{az1:.0f}_{el:.0f}"):
		utils.mkdir(dirname)
		np.savetxt(dirname + "/dets.txt", self.dets, fmt="%5d")
		with open(dirname + "/info.txt", "w") as f:
			for pid, (pattern, m) in enumerate(zip(self.patterns, self.maps)):
				oname = ("scan_" + fmt + ".fits").format(pid=pid, el=pattern[0,0]/utils.degree, az0=pattern[0,1]/utils.degree, az1=pattern[1,1]/utils.degree)
				enmap.write_map(dirname + "/" + oname, m)
				f.write("%8.3f %8.3f %8.3f %s\n" % (pattern[0,0]/utils.degree, pattern[0,1]/utils.degree,
						pattern[1,1]/utils.degree, oname))
	@staticmethod
	def zeros(patterns, dets, res=1*utils.arcmin, det_unit=1, hysteresis=True, dtype=np.float64):
		maps = []
		ndet = len(dets)
		for pattern in patterns:
			az0, az1 = utils.widen_box(pattern, margin=2*res, relative=False)[:,1]
			naz = int(np.ceil((az1-az0)/res))
			az1 = az0 + naz*res
			shape, wcs = enmap.geometry(pos=[[0,az0],[float(ndet)/det_unit*utils.degree,az1]], shape=(ndet,naz), proj="car")
			if hysteresis: shape = (2,)+tuple(shape)
			maps.append(enmap.zeros(shape, wcs, dtype=dtype))
		return PhaseMap(patterns, dets, maps)
	def copy(self):
		patterns = np.array(self.patterns)
		dets     = np.array(self.dets)
		maps     = [m.copy() for m in self.maps]
		return PhaseMap(patterns, dets, maps)

class SignalPhase(Signal):
	def __init__(self, scans, areas, pids, comm, cuts=None, name="phase", ofmt="{name}", output=True, ext="fits"):
		Signal.__init__(self, name, ofmt, output, ext)
		self.areas     = areas # template PhaseMaps defining scan patterns and pixelizations
		self.pids      = pids  # index of each scan into scanning patterns
		self.comm      = comm
		self.cuts      = cuts
		self.dtype     = areas.maps[0].dtype
		self.data = {}
		# Argh! I should hever have switched to string-based detector names!
		def get_uids(detnames): return np.char.partition(detnames, "_")[:,2].astype(int)
		# Set up the detector mapping for each scan
		for pid, scan in zip(pids,scans):
			dets = get_uids(scan.dets)
			inds = utils.find(areas.dets, dets)
			mat  = pmat.PmatScan(scan, areas.maps[pid], inds)
			self.data[scan] = [pid, mat]
		self.dof = zipper.MultiZipper([
			zipper.ArrayZipper(map, comm=comm) for map in self.areas.maps],
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
	def zeros(self, mat=False):
		return [area*0 for area in self.areas.maps]
	def write(self, prefix, tag, ms):
		if not self.output: return
		if self.comm.rank != 0: return
		omaps = self.areas.copy()
		omaps.maps = ms
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s" % (prefix, oname, tag)
		omaps.write(oname)

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

class SignalGain(Signal):
	def __init__(self, scans, signals, dtype, comm, model=None, name="gain", ofmt="{name}_{rank:02}", output=True):
		Signal.__init__(self, name, ofmt, output, ext="txt")
		self.data    = {}
		self.dtype   = dtype
		self.signals = signals
		# The sky model to assume. Must be compatible with the signals
		# passed in. This can be freely updated by assigning to .model
		self.model   = model
		# Calculate the gain degrees of freedom mapping
		self.data    = {}
		i1 = 0
		for si, scan in enumerate(scans):
			i2 = i1+scan.ndet
			self.data[scan] = (i1,i2)
			i1 = i2
		self.ntot    = i1
		self.dof     = zipper.ArrayZipper(np.zeros(self.ntot, self.dtype), shared=False, comm=comm)
	def precompute(self, scan):
		self.profile = np.zeros((scan.ndet,scan.nsamp),self.dtype)
		self.signals.forward(scan, self.profile, self.model)
	def free(self):
		self.profile = None
	def forward(self, scan, tod, gains):
		if scan not in self.data: return
		i1,i2 = self.data[scan]
		tod  += self.profile*gains[i1:i2,None]
	def backward(self, scan, tod, gains):
		if scan not in self.data: return
		i1,i2 = self.data[scan]
		gains[i1:i2] += np.sum(self.profile*tod,1)
	def zeros(self, mat=False): return np.zeros(self.ntot, self.dtype)
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name, rank=self.dof.comm.rank)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		# We want to output the tods in sorted order
		scans = [scan for scan in self.data]
		ids   = [scan.id for scan in scans]
		order = np.argsort(ids)
		from enact import actdata
		with open(oname, "w") as ofile:
			for ind in order:
				scan  = scans[ind]
				i1, i2= self.data[scan]
				gvals = m[i1:i2]
				dets  = actdata.split_detname(scan.dets)[1]
				msg   = [scan.id]
				for det,gval in zip(dets, gvals):
					msg.append("%s:%.3f" % (det,gval))
				msg   = " ".join(msg)
				ofile.write(msg + "\n")

# Special source handling v2:
# Solve for the map and the local model errors around strong sources jointly. The
# equation system will be degenerate, so the map will not contain a complete solution
# in these regions. So before outputting, we need do do
# tod = Pmap map + Psrcsamp srcssamps
# totmap = map_white(tod)
# So this requires a new signal that's very similar to our cut signal, as well as a
# postprocessing operation that adds up the values in the src pixels.

class SignalSrcSamp(SignalCut):
	def __init__(self, scans, dtype, comm, mask=None, name="srcsamp", ofmt="{name}_{rank:02}", output=False, cut_type="full", sys="cel", keep=True):
		Signal.__init__(self, name, ofmt, output, ext="hdf")
		self.data  = {}
		self.dtype = dtype
		self.keep  = keep
		self.sys   = sys
		self.cut_type = cut_type
		cutrange = [0,0]
		self.mask  = mask
		for scan in scans:
			# Project our sample density into time domain
			tod      = np.zeros((scan.ndet,scan.nsamp), np.float32)
			pmap     = pmat.PmatMap(scan, mask, sys=sys, order=1)
			pmap.forward(tod, mask)
			tod[:]   = tod > 0.5
			# Accumulate density and use this to define a sample mask
			cumsamps = np.concatenate([[0],utils.floor(np.cumsum(tod))])
			# minimum shouldn't be necessary, but present to guard against floating point issues
			tod_mask = np.minimum(cumsamps[1:]-cumsamps[:-1],1).reshape(tod.shape)
			del cumsamps, tod
			cut = sampcut.from_mask(tod_mask); del tod_mask
			mat = pmat.PmatCut(scan, cut_type, tmul=keep*1.0, cut=cut)
			cutrange = [cutrange[1], cutrange[1]+mat.njunk]
			self.data[scan] = [mat, cutrange]
		self.njunk = cutrange[1]
		self.dof = zipper.ArrayZipper(np.zeros(self.njunk, self.dtype), shared=False, comm=comm)

class SignalTemplate(Signal):
	def __init__(self, scans, template, comm, name="template", ofmt="{name}", output=True,
			ext="txt", pmat_order=None, sys=None, extra=[]):
		Signal.__init__(self, name, ofmt, output, ext)
		self.template = template
		self.dof  = zipper.ArrayZipper(np.zeros(len(scans), dtype=template.dtype), shared=False, comm=comm)
		self.dtype= template.dtype
		self.data = {scan: pmat.PmatMap(scan, template, order=pmat_order, sys=sys, extra=extra) for scan in scans}
		self.inds = {scan:i for i, scan in enumerate(scans)}
		self.scans= scans
		self.n    = len(scans)
	def forward(self, scan, tod, work, tmul=1, mmul=1):
		if scan not in self.data: return
		self.data[scan].forward(tod, self.template*work[self.inds[scan]], tmul=tmul, mmul=mmul)
	def backward(self, scan, tod, work, tmul=1, mmul=1):
		if scan not in self.data: return
		Ptd = self.template*0
		self.data[scan].backward(tod, Ptd, tmul=tmul)
		work[self.inds[scan]] = work[self.inds[scan]]*mmul + np.sum(Ptd)
	def zeros(self, mat=False): return np.zeros(self.n, self.dtype)
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name, rank=self.dof.comm.rank)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		with open(oname, "w") as ofile:
			for i in range(self.n):
				ofile.write("%s %8.3f\n" % (self.scans[i].id, m[i]))

class Signals(Signal):
	"""Class for handling multiple signals jointly."""
	def __init__(self, signals, comm, name="signals", ofmt="{name}_{rank:02}", ext="hdf", output=False):
		Signal.__init__(self, name, ofmt, output, ext)
		self.signals = signals
		self.dof     = zipper.MultiZipper([signal.dof for signal in signals], comm=comm)
		self.precon  = PreconSignals(signals)
	def zeros(self, mat=False): return [signal.zeros(mat=mat) for signal in self.signals]
	def prepare(self, maps): return [signal.prepare(map) for signal,map in zip(self.signals, maps)]
	def filter(self, maps): return [signal.filter(map) for signal,map in zip(self.signals, maps)]
	def precompute(self, scan):
		for signal in self.signals:
			signal.precompute(scan)
	def forward (self, scan, tod, works):
		for signal, work in list(zip(self.signals, works))[::-1]:
			signal.forward(scan, tod, work)
	def backward(self, scan, tod, works):
		for signal, work in zip(self.signals, works):
			signal.backward(scan, tod, work)
	def free(self):
		for signal in self.signals:
			signal.free()
	def finish  (self, maps, works):
		for signal, map, work in zip(self.signals, maps, works):
			signal.finish(map, work)
	def transform(self, maps, op): return [signal.transform(map, op) for signal,map in zip(self.signals, maps)]
	def polinv(self, maps): return [signal.polinv(map) for signal,map in zip(self.signals, maps)]
	def polmul(self, mats, maps, inplace=False): return [signal.polmul(mat, map, inplace=inplace) for signal,mat,map in zip(self.signals, mats, maps)]
	# This one is a potentially cheaper version of self.prepare(self.zeros())
	def write   (self, prefix, tag, maps):
		for signal, map in zip(self.signals, maps):
			signal.write(prefix, tag, map)
	def postprocess(self, maps): return [signal.postprocess(map) for signal,map in zip(self.signals, maps)]
	def prior(self, scans, xins, xouts):
		for signal, scans, xin, xout in zip(self.signals, xins, xouts):
			signal.prior(scans, xin, xout)

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
	def __init__(self, delayed=False): pass
	def build(self, scans): pass
	def __call__(self, m): pass
	def write(self, prefix="", ext="fits"): pass

config.default("eig_limit", 1e-3, "Smallest relative eigenvalue to invert in eigenvalue inversion. Ones smaller than this are set to zero.")
# The binned preconditioners use a 3x3 covmat per pixel. This matrix is sometimes
# poorly conditioned, and can't be inverted. I currently use eigenvalue inversion
# for this, which works by setting bad eigenvalues to zero. But is this sufficient?
# Consider a pixel that is hit by a single sample with phase [1,1,0], resulting in
# the covmat [110,110,000]. This has a single nonzero eigenvalue, and after
# eigenvalue inversion it will still be proportional to [110,110,000]. This preconditioner
# forces U to be zero, but it allows T,Q to float. This means that we can't distinguish
# [0,0,0] from [1,-1,0] or [1e9,-1e9,0], which can lead to runaway pixel values.
# To avoid this problem we can make the poorly conditioned pixels T-only by setting
# all but [0,0] of icov to 0.
#
# I've moved to this workaround, even though it appears to perform marginally worse
# at for my one-tod 10-cg test. The main thing that helps in both cases is to use an
# eig_limit like 1e-3 instead of 1e-6, which I used before

# Preconditioners now have a build(self, scans) method that does the actual building.
# To keep backward compatibility this is called automatically in the constructor
# unless delayed=True is passed (ideally this would default to True, but that would
# break things, so I don't do that right away). build() should be safe to call multiple
# times, which would update the preconditioners based on the scans without leaving it
# in an inconsistent state

class PreconMapBinned:
	def __init__(self, signal, scans, weights, noise=True, hits=True, mask=None, delayed=False):
		"""Binned preconditioner: (P'W"P)", where W" is a white
		nosie approximation of N". If noise=False, instead computes
		(P'P)". If hits=True, also computes a hitcount map."""
		self.signal = signal
		self.weights= weights
		self.noise  = noise
		self.mask   = mask
		self.calc_hits = hits
		# Backwards compatibility
		if not delayed: self.build(scans)
	def build(self, scans):
		self.div = self.signal.zeros(mat=True)
		calc_div_map(self.div, self.signal, scans, self.weights, noise=self.noise)
		if self.mask is not None: self.div *= self.mask
		self.idiv = array_ops.eigpow(self.div, -1, axes=[-4,-3], lim=config.get("eig_limit"), fallback="scalar")
		if self.calc_hits:
			# Build hitcount map too
			self.hits = self.signal.area.copy()
			self.hits = calc_hits_map(self.hits, self.signal, scans)
			if self.mask is not None: self.hits *= self.mask
		else: self.hits = None
	def __call__(self, m):
		m[:] = array_ops.matmul(self.idiv, m, axes=[-4,-3])
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)
		if self.hits is not None:
			self.signal.write(prefix, "hits", self.hits)

class PreconDmapBinned:
	def __init__(self, signal, scans, weights, noise=True, hits=True, delayed=False):
		"""Binned preconditioner: (P'W"P)", where W" is a white
		nosie approximation of N". If noise=False, instead computes
		(P'P)". If hits=True, also computes a hitcount map."""
		self.signal  = signal
		self.weights = weights
		self.noise   = noise
		self.calc_hits = hits
		if not delayed: self.build(scans)
	def build(self, scans):
		self.div = self.signal.zeros(mat=True)
		calc_div_map(self.div, self.signal, scans, self.weights, noise=self.noise)
		self.idiv = self.div.copy()
		for dtile in self.idiv.tiles:
			dtile[:] = array_ops.eigpow(dtile, -1, axes=[-4,-3], lim=config.get("eig_limit"), fallback="scalar")
		if self.calc_hits:
			# Build hitcount map too
			self.hits = self.signal.area.copy()
			self.hits = calc_hits_map(self.hits, self.signal, scans)
		else: self.hits = None
	def __call__(self, m):
		for idtile, mtile in zip(self.idiv.tiles, m.tiles):
			mtile[:] = array_ops.matmul(idtile, mtile, axes=[-4,-3])
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)
		if self.hits is not None:
			self.signal.write(prefix, "hits", self.hits)

class PreconMapHitcount:
	def __init__(self, signal, scans, delayed=False):
		"""Hitcount preconditioner: (P'P)_TT."""
		self.signal = signal
		if not delayed: self.build(scans)
	def build(self, scans):
		self.hits = self.signal.area.copy()
		self.hits = calc_hits_map(self.hits, self.signal, scans)
		self.ihits= 1.0/np.maximum(self.hits,1)
	def __call__(self, m):
		m *= self.ihits[...,None,:,:]
	def write(self, prefix):
		self.signal.write(prefix, "hits", self.hits)

class PreconDmapHitcount:
	def __init__(self, signal, scans, delayed=False):
		"""Hitcount preconditioner: (P'P)_TT"""
		self.signal = signal
		if not delayed: self.build(scans)
	def build(self, scans):
		self.hits = self.signal.area.copy()
		self.hits = calc_hits_map(self.hits, self.signal, scans)
	def __call__(self, m):
		for htile, mtile in zip(self.hits.tiles, m.tiles):
			htile = np.maximum(htile, 1)
			mtile /= htile[...,None,:,:]
	def write(self, prefix):
		self.signal.write(prefix, "hits", self.hits)

class PreconMapTod:
	def __init__(self, signal, scans, weights, delayed=False):
		"""Preconditioner based on inverting the tod noise model independently per TOD,
		along with a P'P inversion: precon = P"'NP", where P" = P(P'P)" """
		self.signal   = signal
		self.weights  = weights
		self.scans    = scans
		self.maxnoise = 10
		if not delayed: self.build(scans)
	def build(self, scans):
		# First get P'P, which is the standard div but with no noise model applied
		ncomp = self.signal.area.shape[0]
		self.ptp  = enmap.zeros((ncomp,)+self.signal.area.shape, self.signal.area.wcs, self.signal.area.dtype)
		calc_div_map(self.ptp, self.signal, scans, self.weights, noise=False)
		self.iptp = array_ops.eigpow(self.ptp, -1, axes=[0,1], lim=config.get("eig_limit"))
		self.hits = self.ptp[0,0].astype(np.int32)
	def __call__(self, m):
		# This is pretty slow. Let's see if it is worth it. I fear the discontinuities
		# will be just as slow to resolve
		m[:]    = array_ops.matmul(self.iptp, m, axes=[0,1])
		omap    = m*0
		for scan in self.scans:
			tod  = np.zeros([scan.ndet, scan.nsamp], self.signal.dtype)
			self.signal.precompute(scan)
			self.signal.forward    (scan, tod, m)
			for weight in self.weights:       weight(scan, tod)
			noise_ref = np.median(scan.noise.D)*self.maxnoise
			scan.noise.D = np.minimum(scan.noise.D, noise_ref)
			scan.noise.E = np.minimum(scan.noise.E, noise_ref)
			scan.noise.apply(tod, inverse=True)
			for weight in self.weights[::-1]: weight(scan, tod)
			sampcut.gapfill_const(scan.cut, tod, inplace=True)
			self.signal.backward(scan, tod, omap)
			self.signal.free()
		m[:] = 0
		self.signal.finish(m, omap)
		m[:] = array_ops.matmul(self.iptp, m, axes=[0,1])
		return m
	def write(self, prefix):
		self.signal.write(prefix, "ptp", self.ptp)
		self.signal.write(prefix, "hits", self.hits)

class PreconCut:
	def __init__(self, signal, scans, weights=[], delayed=False):
		self.signal  = signal
		self.weights = weights
		if not delayed: self.build(scans)
	def build(self, scans):
		# I'm ignoring weights in this preconditioner after the performance
		# invesstigation below:
		# With 200 steps max:
		# full+white+nowin: super  1e-40 @ CG 3
		# full+white+igwin: good   1e-12
		# full+white+win:   super  1e-35 @ CG 9
		# full+jon  +nowin: medium 1e-7
		# full+jon  +igwin: medium 1e-7
		# full+jon  +win:   good   1e-11
		# poly+white+nowin: super  1e-32 @ CG 16
		# poly+white+igwin: good   1e-11
		# poly+white+win:   great  1e-31 @ CG 162
		# poly+jon  +nowin: medium 1e-8
		# poly+jon  +igwin: medium 1e-8
		# poly+jon  +win:   bad    1e-4
		# So overall that's bad is anything with poly+win.
		# The best would be to not do windowing. I'll consider that. But if one
		# has windowing, ignoring it in the preconditioner seems best. The issue
		# is that windowing breaks the orthogonality of the legendre polynomials,
		# and the preconditioner relies on that orthogonality. Ignoring the windowing
		# keeps the preconditioner self-consistent even though it's no longer fully
		# consistent with the actual data.
		junk  = self.signal.zeros()
		iwork = self.signal.prepare(junk)
		owork = self.signal.prepare(junk)
		iwork[:] = 1
		for scan in scans:
			with bench.mark("div_" + self.signal.name):
				tod = np.zeros((scan.ndet, scan.nsamp), iwork.dtype)
				self.signal.forward (scan, tod, iwork)
				#for weight in self.weights: weight(scan, tod)
				scan.noise.white(tod)
				#for weight in self.weights[::-1]: weight(scan, tod)
				self.signal.backward(scan, tod, owork)
			times = [bench.stats[s]["time"].last for s in ["div_"+self.signal.name]]
			L.debug("div %s %6.3f %s" % ((self.signal.name,)+tuple(times)+(scan.id,)))
		self.signal.finish(junk, owork)
		junk = np.abs(junk)
		# Avoid negative numbers or division by zero. Should not be necessary
		# if we ignore windowing, but doesn't hurt much.
		ref  = np.max(junk)/1e7 if len(junk) > 0 else 1
		self.idiv = 1/np.maximum(junk,ref)
	def __call__(self, m):
		m *= self.idiv
	def write(self, prefix):
		self.signal.write(prefix, "idiv", self.idiv)

class PreconPhaseBinned:
	def __init__(self, signal, scans, weights, delayed=False):
		self.signal  = signal
		self.weights = weights
		if not delayed: self.build(scans)
	def build(self, scans):
		div = [area*0+1 for area in self.signal.areas.maps]
		# The cut samples are included here becuase they must be avoided, but the
		# actual computation of the junk sample preconditioner happens elsewhere.
		# This is a bit redundant, but should not cost much time since this only happens
		# in the beginning.
		iwork = self.signal.prepare(div)
		owork = self.signal.prepare(self.signal.zeros())
		prec_div_helper(self.signal, scans, self.weights, iwork, owork)
		self.signal.finish(div, owork)
		hits = self.signal.zeros()
		owork = self.signal.prepare(hits)
		for scan in scans:
			tod = np.full((scan.ndet, scan.nsamp), 1, self.signal.dtype)
			self.signal.backward(scan, tod, owork)
		self.signal.finish(hits, owork)
		for i in range(len(hits)):
			hits[i] = hits[i].astype(np.int32)
		self.div = div
		self.hits = hits
	def __call__(self, ms):
		for d, m in zip(self.div, ms):
			m[d!=0] /= d[d!=0]
	def write(self, prefix):
		self.signal.write(prefix, "div", self.div)

class PreconSignals:
	def __init__(self, signals, delayed=False):
		self.signals = signals
		# If we're not delayed, then our signals' precons will already
		# have been initialized, so don't do anything in either case
	def build(self, scans):
		for signal in self.signals:
			signal.precon.build(scans)
	def __call__(self, maps):
		for signal, map in zip(self.signals, maps):
			signal.precon(map)
	def write(self, prefix="", ext=None):
		for signal in self.signals:
			signal.write(prefix=prefix)

def prec_div_helper(signal, scans, weights, iwork, owork, cuts=None, noise=True):
	# The argument list of this one is so long that it almost doesn't save any
	# code.
	if cuts is None: cuts = [scan.cut for scan in scans]
	for si, scan in enumerate(scans):
		with bench.mark("div_Pr_" + signal.name):
			signal.precompute(scan)
		with bench.mark("div_P_" + signal.name):
			tod = np.zeros((scan.ndet, scan.nsamp), signal.dtype)
			signal.forward(scan, tod, iwork)
			#signal_cut.forward (scan, tod, ijunk)
		with bench.mark("div_white"):
			for weight in weights: weight(scan, tod)
			if noise: scan.noise.white(tod)
			for weight in weights[::-1]: weight(scan, tod)
		with bench.mark("div_PT_" + signal.name):
			#signal_cut.backward(scan, tod, ojunk)
			sampcut.gapfill_const(cuts[si], tod, inplace=True)
			signal.backward(scan, tod, owork)
		with bench.mark("div_Fr_" + signal.name):
			signal.free()
		times = [bench.stats[s]["time"].last for s in ["div_P_"+signal.name, "div_white", "div_PT_" + signal.name]]
		L.debug("div %s %6.3f %6.3f %6.3f %s" % ((signal.name,)+tuple(times)+(scan.id,)))

def calc_div_map(div, signal, scans, weights, cuts=None, noise=True):
	# The cut samples are included here becuase they must be avoided, but the
	# actual computation of the junk sample preconditioner happens elsewhere.
	# This is a bit redundant, but should not cost much time since this only happens
	# in the beginning.
	for i in range(div.shape[-4]):
		div[...,i,i,:,:] = 1
		iwork = signal.prepare(div[...,i,:,:,:])
		owork = signal.prepare(signal.zeros())
		prec_div_helper(signal, scans, weights, iwork, owork, cuts=cuts, noise=noise)
		signal.finish(div[...,i,:,:,:], owork)

def calc_crosslink_map(signal, scans, weights, cuts=None, noise=True):
	saved_comps = [scan.comps.copy() for scan in scans]
	if cuts is None: cuts = [scan.cut for scan in scans]
	for scan in scans: scan.comps[:] = np.array([1,1,0])
	cmap    = signal.zeros()
	owork   = signal.prepare(cmap)
	for si, scan in enumerate(scans):
		with bench.mark("cmap_Pr_" + signal.name): signal.precompute(scan)
		with bench.mark("cmap_tod"):
			tod = np.full((scan.ndet, scan.nsamp), 1.0, signal.dtype)
		with bench.mark("cmap_white"):
			for weight in weights: weight(scan, tod)
			if noise: scan.noise.white(tod)
			for weight in weights: weight(scan, tod)
		with bench.mark("cmap_PT_" + signal.name):
			#signal_cut.backward(scan, tod, ojunk)
			sampcut.gapfill_const(cuts[si], tod, inplace=True)
			signal.backward(scan, tod, owork)
		signal.free()
		times = [bench.stats[s]["time"].last for s in ["cmap_white", "cmap_PT_" + signal.name]]
		L.debug("crossmap %s %6.3f %6.3f %s" % ((signal.name,)+tuple(times)+(scan.id,)))
	signal.finish(cmap, owork)
	# Restore saved components
	for scan, comp in zip(scans, saved_comps): scan.comps = comp
	return cmap

def calc_ptsrc_map(signal, scans, src_filters):
	# First compute P'W"srcs
	cmap    = signal.zeros()
	owork   = signal.prepare(cmap)
	for scan in scans:
		tod = np.zeros((scan.ndet, scan.nsamp), signal.dtype)
		with bench.mark("srcmap_srcs"):
			for src_filter in src_filters:
				src_filter(scan, tod)
		with bench.mark("srcmap_PT_" + signal.name):
			sampcut.gapfill_const(scan.cut, tod, inplace=True)
			signal.backward(scan, tod, owork)
		signal.free()
		times = [bench.stats[s]["time"].last for s in ["srcmap_srcs", "srcmap_PT_" + signal.name]]
		L.debug("srcmap %s %6.3f %6.3f %s" % ((signal.name,)+tuple(times)+(scan.id,)))
	signal.finish(cmap, owork)
	# Then divide out the hits
	with utils.nowarn():
		cmap /= signal.precon.hits
	cmap.fillbad(0, inplace=True)
	# We want to know what the source model is, which is the opposite of what was
	# subtracted
	cmap *= -1
	return cmap

def calc_icov_map(signal, scans, pos, weights):
	"""Compute a map containing the inverse covariance structure around the set
	of positions pos[:,{y,x}] in pixels. This will be computed in one
	operation, so if the points are too close to each other their covariance
	information will interfere with each other."""
	icov   = signal.zeros()
	ocov   = icov.copy()
	# Set the chosen positions to one. This is very inelegant. Should have
	# global pixel setting methods in dmap
	pos    = np.zeros([1,2],int)+pos # broadcast to correct shape
	if isinstance(icov, enmap.ndmap):
		icov[0,pos[:,0],pos[:,1]] = 1
	elif isinstance(icov, dmap.Dmap):
		for tile, ind in zip(icov.tiles, icov.loc_inds):
			pbox = icov.geometry.tile_boxes[ind]
			for pix in pos:
				y,x = pix - pbox[0]
				if y >= 0 and x >= 0 and y < tile.shape[-2] and x < tile.shape[-1]:
					tile[0,y,x] = 1
	# The rest proceeds similarly to the crosslinking map
	iwork   = signal.prepare(icov)
	owork   = signal.prepare(ocov)
	for scan in scans:
		with bench.mark("icov_Pr_" + signal.name): signal.precompute(scan)
		with bench.mark("icov_P_" + signal.name):
			tod = np.zeros((scan.ndet, scan.nsamp), signal.dtype)
			signal.forward(scan, tod, iwork)
		with bench.mark("icov_nmat"):
			for weight in weights: weight(scan, tod)
			scan.noise.apply(tod)
			for weight in weights: weight(scan, tod)
		with bench.mark("icov_PT_" + signal.name):
			sampcut.gapfill_const(scan.cut, tod, inplace=True)
			signal.backward(scan, tod, owork)
		with bench.mark("icov_Fr_" + signal.name): signal.free()
		times = [bench.stats[s]["time"].last for s in ["icov_P_" + signal.name, "icov_nmat", "icov_PT_" + signal.name]]
		L.debug("icov %s %6.3f %6.3f %6.3f %s" % ((signal.name,)+tuple(times)+(scan.id,)))
	signal.finish(ocov, owork)
	return ocov[0]

def calc_hits_map(hits, signal, scans, cuts=None):
	hits = hits*0
	work = signal.prepare(hits)
	if cuts is None: cuts = [scan.cut for scan in scans]
	for si, scan in enumerate(scans):
		with bench.mark("hits_Pr_" + signal.name):
			signal.precompute(scan)
		with bench.mark("hits_PT"):
			tod = np.full((scan.ndet, scan.nsamp), 1, hits.dtype)
			sampcut.gapfill_const(cuts[si], tod, inplace=True)
			signal.backward(scan, tod, work)
		with bench.mark("hits_Fr_" + signal.name):
			signal.free()
		times = [bench.stats[s]["time"].last for s in ["hits_PT"]]
		L.debug("hits %s %6.3f %s" % ((signal.name,)+tuple(times)+(scan.id,)))
	with bench.mark("hits_reduce"):
		signal.finish(hits, work)
	return hits[...,0,:,:].astype(np.int32)

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

class PriorSrcSamp:
	def __init__(self, signal_srcsamp, eps_edge=10, eps_core=0.01, redge=2*utils.arcmin):
		# Measure the distance of each sample from the edge.
		# The mask can be quite low-res, so we will need to interpolate. We already have
		# bilinear mapmaking available for this.
		dtype     = signal_srcsamp.dtype
		distmap   = signal_srcsamp.mask*0
		distmap[0]= signal_srcsamp.mask.preflat[0].distance_transform()
		distsamps = signal_srcsamp.zeros()
		whitesamps= signal_srcsamp.zeros()
		assert signal_srcsamp.cut_type == "full", "PriorSrcSamp requires per-sampe (full) cuts in SignalSrcSamp"
		for scan in signal_srcsamp.data:
			tod  = np.zeros([scan.ndet, scan.nsamp], dtype)
			pmap = pmat.PmatMap(scan, distmap, sys=signal_srcsamp.sys, order=1)
			pmap.forward(tod, distmap)
			# This assumes that we use per-sample degrees of freedom in SignalSrcSamp, which
			# is currently hardcoded, so this is safe until that changes.
			signal_srcsamp.backward(scan, tod, distsamps)
			# white noise level, to have something sensible to scale the prior with
			tod[:] = 1
			scan.noise.white(tod)
			signal_srcsamp.backward(scan, tod, whitesamps)
		# Use these distances to build the per-sample prior amplitude
		x = np.minimum(distsamps/redge, 1)
		self.epsilon  = np.exp(np.log(eps_edge) * (1-x) + np.log(eps_core) * x)
		self.epsilon *= whitesamps
	def __call__(self, scans, imap, omap):
		omap += imap * self.epsilon

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

class FilterNull:
	def __call__(self, scan, tod): pass

class FilterScale:
	def __init__(self, scale): self.scale = scale
	def __call__(self, scan, tod): tod *= self.scale

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
	def __init__(self, scans, signal_map, signal_cut, prec_ptp, daz=None, nt=None, weighted=False, deslope=False):
		self.scans = scans
		self.signal_map = signal_map
		self.signal_cut = signal_cut
		self.daz, self.nt = daz, nt
		self.ptp = prec_ptp
		self.weighted = weighted
		self.deslope  = deslope
	def __call__(self, imap):
		return self.postfilter_TP_separately(imap)
	def postfilter_TP_separately(self, imap):
		res = imap*0
		tmp = imap.copy(); tmp[1:] = 0; res[:1] = self.postfilter_map(tmp)[:1]
		tmp = imap.copy(); tmp[:1] = 0; res[1:] = self.postfilter_map(tmp)[1:]
		return res
	def postfilter_map(self, imap):
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
			# Skip cut when going forwards - we don't want to introduce lots
			# of sudden jumps to zero in our simulated tod. We do this by
			# writing [:0:-1] instead of [::-1], since the 0th entry is the cut.
			# This could cause problems if a tod both starts outside our hit
			# area and has cuts there, though...
			with bench.mark("post_P"):
				for signal, work in zip(signals, iwork)[:0:-1]:
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
			with bench.mark("post_F"):
				todfilter.filter_poly_jon(tod, scan.boresight[:,1], weights=weights, naz=naz, nt=self.nt, deslope=self.deslope)
			# Here we include the cuts. This isn't strictly necessary, but it
			# preserves our hit area
			with bench.mark("post_PT"):
				for signal, work in zip(signals, owork):
					signal.backward(scan, tod, work)
				if self.weighted: del weights, tod
			times = [bench.stats[s]["time"].last for s in ["post_P","post_F","post_PT"]]
			L.debug("post P %5.3f N %5.3f P' %5.3f %s %4d" % (tuple(times)+(scan.id,scan.ndet)))
		for signal, map, work in zip(signals, omaps, owork):
			signal.finish(map, work)
		# Must use (P'P)" here, not any other preconditioner!
		self.ptp(omaps[1])
		return omaps[1]

class FilterAddMap:
	def __init__(self, scans, map, sys=None, mul=1, tmul=1, pmat_order=None, ptoff=None, det_muls=None, poleffs=None, tconstoff=None):
		self.map, self.sys, self.mul, self.tmul, self.ptoff, self.det_muls, self.poleffs, self.tconstoff = map, sys, mul, tmul, ptoff, det_muls, poleffs, tconstoff
		self.data = {}
		for scan in scans:
			# Make a shallow copy of the scan where we can override properties for the
			# purpose simulating instrumental errors
			myscan = scan.copy(shallow=True)
			myscan.comps   = scan.comps.copy()
			myscan.offsets = scan.offsets.copy()
			if det_muls is not None:
				myscan.comps *= det_muls[scan][:,None]
			if poleffs is not None:
				myscan.comps[:,1:] *= poleffs[scan][:,None]
			if ptoff is not None:
				myscan.offsets[:,1:] += ptoff[scan]
			pmap = pmat.PmatMap(myscan, map, order=pmat_order, sys=sys)
			self.data[scan] = pmap
	def __call__(self, scan, tod):
		pmat = self.data[scan]
		pmat.forward(tod, self.map, tmul=self.tmul, mmul=self.mul)
		# optionally apply time constant error
		if self.tconstoff is not None:
			assert self.tmul == 0, "FilterAddMap requires tmul=0 when injecting a signal with time constant erorrs"
			from enact import filters
			ft    = fft.rfft(tod)
			freq  = fft.rfftfreq(tod.shape[1], 1/scan.srate)
			for di in range(len(ft)):
				ft[di] /= filters.tconst_filter(freq, self.tconstoff[scan][di])
			fft.irfft(ft, tod, normalize=True)

class FilterAddDmap:
	def __init__(self, scans, subinds, dmap, sys=None, mul=1, tmul=1, pmat_order=None):
		self.map, self.sys, self.mul, self.tmul = dmap, sys, mul, tmul
		self.data = {}
		work = dmap.tile2work()
		for scan, subind in zip(scans, subinds):
			self.data[scan] = [pmat.PmatMap(scan, work.maps[subind], order=pmat_order, sys=sys), work.maps[subind]]
	def __call__(self, scan, tod):
		pmat, work = self.data[scan]
		pmat.forward(tod, work, tmul=self.tmul, mmul=self.mul)

def extra_ptoff(ptoff):
	if ptoff is None or np.all(ptoff == 0): return []
	else: return [lambda pos,time: np.concatenate([pos[:2]+ptoff[:,None],pos[2:]],0)]

class PostAddMap:
	# This one is easy if imap and map are compatible (the common case),
	# but hard otherwise, as it requires reprojection in that case. We
	# only support the compatible case for now.
	def __init__(self, map, mul=1):
		self.map, self.mul = map, mul
	def __call__(self, imap):
		return imap + self.map*self.mul

def prepare_buddy_map(map, rcut=1*utils.arcmin, rapod=5*utils.arcmin):
	"""Prepare input map for buddy subtraction. Cut areas within rcut
	radians from an unhit area, and apodize a further rapod from here.
	Areas outside the map geometry are not counted as unhit, so the
	edge of the geometry won't be cut or apodized.

	The goal of this operation is to avoid having very noisy areas near
	the edge leak into the useful part of the map due to buddy subtraction."""
	mask = map.preflat[0] != 0
	mask = mask.distance_transform(rmax=rcut) >= rcut
	apod = enmap.apod_mask(mask, width=rapod, edge=False).astype(map.dtype)
	return map*apod

class FilterBuddy:
	def __init__(self, scans, map, sys=None, mul=1, tmul=1, pmat_order=None,
			rcut=1*utils.arcmin, rapod=5*utils.arcmin):
		self.map, self.sys, self.mul, self.tmul = map, sys, mul, tmul
		# Handle potentially super-noisy edge
		self.map = prepare_buddy_map(self.map, rcut=rcut, rapod=rapod)
		self.data = {scan: pmat.PmatMapMultibeam(scan, map, scan.buddy_offs,
			scan.buddy_comps, order=pmat_order, sys=sys) for scan in scans}
	def __call__(self, scan, tod):
		pmat = self.data[scan]
		pmat.forward(tod, self.map, tmul=self.tmul, mmul=self.mul)

class FilterBuddyDmap:
	def __init__(self, scans, subinds, dmap, sys=None, mul=1, tmul=1, pmat_order=None,
			rcut=1*utils.arcmin, rapod=5*utils.arcmin):
		self.map, self.sys, self.mul, self.tmul = dmap.copy(), sys, mul, tmul
		# Handle potentially super-noisy edge
		for tile in self.map.tiles:
			tile[:] = prepare_buddy_map(tile, rcut=rcut, rapod=rapod)
		self.data = {}
		work = self.map.tile2work()
		for scan, subind in zip(scans, subinds):
			self.data[scan] = [pmat.PmatMapMultibeam(scan, work.maps[subind], scan.buddy_offs,
				scan.buddy_comps, order=pmat_order, sys=sys), work.maps[subind]]
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
			signal_map.precon = PreconMapBinned(signal_map, [scan], [], noise=self.prec=="bin", hits=False)
		else: raise NotImplementedError
		# Ok, we can now solve the system
		cg = CG(eqsys.A, eqsys.b, M=eqsys.M, dot=eqsys.dot, destroy_b=True)
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
	def __init__(self, scans, params, sys=None, mul=1, tmul=1):
		self.params = params
		self.data = {}
		for scan in scans:
			self.data[scan] = pmat.PmatPtsrc(scan, params, sys=sys, pmul=mul, tmul=tmul)
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
		nmode = int(tod.shape[-1]//2*self.fit_highpass/scan.srate)
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
	def __init__(self, basic=False):
		self.basic = basic
	def __call__(self, scan, tod):
		if self.basic: gapfill.gapfill(tod, scan.cut_basic, inplace=True)
		else:          gapfill.gapfill(tod, scan.cut, inplace=True)

class FilterDeslope:
	def __call__(self, scan, tod):
		tod[:] = utils.deslope(tod)

class FilterAddPhase:
	def __init__(self, scans, phasemap, pids, mmul=1, tmul=1):
		self.phasemap = phasemap
		self.tmul     = tmul
		for i,m in enumerate(self.phasemap.maps):
			self.phasemap.maps[i] = m*mmul
		self.data = {}
		# Argh! I should hever have switched to string-based detector names!
		def get_uids(detnames): return np.char.partition(detnames, "_")[:,2].astype(int)
		for pid, scan in zip(pids, scans):
			dets = get_uids(scan.dets)
			inds = utils.find(phasemap.dets, dets)
			mat  = pmat.PmatScan(scan, phasemap.maps[pid], inds)
			self.data[scan] = [pid, mat]
	def __call__(self, scan, tod):
		pid, mat = self.data[scan]
		if self.tmul != 1: tod *= self.tmul
		mat.forward(tod, self.phasemap.maps[pid])

class FilterDeprojectPhase:
	def __init__(self, scans, phasemap, pids, perdet=False, mmul=1, tmul=1):
		print("Warning: FilterDeprojectPhase gives nonsensical fits due to ignoring template noise")
		self.phasemap = phasemap
		self.perdet   = perdet
		self.mmul     = mmul
		self.tmul     = tmul
		self.data = {}
		for pid, scan in zip(pids, scans):
			inds = utils.find(phasemap.dets, scan.dets)
			mat  = pmat.PmatScan(scan, phasemap.maps[pid], inds)
			self.data[scan] = [pid, mat]
	def __call__(self, scan, tod):
		pid, mat = self.data[scan]
		def weight(a):
			fa = fft.rfft(a)
			fa[...,:2000] = 0
			fa[...,fa.shape[-1]//4:] = 0
			fft.irfft(fa,a, normalize=True)
		t  = tod*0
		mat.forward(t, self.phasemap.maps[pid])
		Nt = t.copy()
		# This doesn't work - scan.noise is undefined at this point
		weight(t)
		# Debug dump
		Nd = tod.copy()
		weight(Nd)
		with h5py.File("test_tod.hdf","w") as hfile:
			hfile["d"]   = tod[:16]
			hfile["t"]   = t[:16]
			hfile["Nd"]  = Nd[:16]
			hfile["Nt"]  = Nt[:16]
		with h5py.File("test_phase.hdf","w") as hfile:
			for name, arr in [("d",tod),("t",t),("Nd",Nd),("Nt",Nt)]:
				map = self.phasemap.maps[pid]*0
				mat.backward(arr, map)
				hfile[name] = map

		rhs = np.sum(Nt*tod,1)
		div = np.sum(Nt*t,  1)
		del Nt
		if not self.perdet or True:
			rhs = np.sum(rhs)[None]
			div = np.sum(div)[None]
		amps = rhs/div * self.mmul
		print(np.sort(amps)[::10])
		1/0
		if self.tmul != 1: tod *= self.tmul
		tod -= t * amps[:,None]

class FilterBroadenBeamHor:
	def __init__(self, ibeam, obeam):
		"""Apply a lowpass-filter to the TOD such that the the effective beam
		size in the scanning direction increases from ibeam to obeam, both given
		as standard deviations in arcmin."""
		self.ibeam, self.obeam = ibeam, obeam
	def __call__(self, scan, tod):
		broaden_beam_hor(tod, scan, self.ibeam, self.obeam)

class FilterInpaintSub:
	def __init__(self, targets, fknee=10, alpha=10):
		"""Use the samples more than dist away from each target in the targets string
		to predict what the noise should be doing in the samples closer than that, and
		subtract this prediction from the whole TOD. The prediction is based on gapfill_joneig
		plus an extra smoothing. This filter is appropriate for reducing correlated noise in
		the special case where you know the signal is almost entirely contained in small
		regions near some objects.

		The targets string has the format
		  obj:dist,obj:dist,...
		where obj can either be a capitalized planet name or [ra,dec] in degrees.
		:dist is in degrees, and is optional. If left out the previous dist will be
		repeated. The default is 0.5."""
		self.targets = targets
		self.fknee   = fknee
		self.alpha   = alpha
	def __call__(self, scan, tod):
		from enact import cuts as actcuts
		objs = utils.split_outside(self.targets,",")
		dist = 0.5*utils.degree
		# Reformat point offset to actdata convention, which is what avoidance_cut wants
		bore = scan.boresight.T.copy()
		bore[0] += utils.mjd2ctime(scan.mjd0)
		# Build a "cut" that selects the samples near the objects
		mycuts = sampcut.empty(*tod.shape)
		for obj in objs:
			toks = obj.split(":")
			objname = toks[0]
			if objname.startswith("["):
				objname = [float(w)*utils.degree for w in objname[1:-1].split(",")]
			if len(toks) > 1: dist = float(toks[1])*utils.degree
			mycuts *= actcuts.avoidance_cut(bore, scan.offsets[:,1:], scan.site, objname, dist)
		# Use this to gapfill a copy of the tod
		model = gapfill.gapfill_joneig(tod, mycuts, inplace=False)
		# Smooth this with a simple lowpass filter
		ftod = fft.rfft(model)
		freq = fft.rfftfreq(model.shape[-1])*scan.srate
		flt  = 1/(1+(freq/self.fknee)**self.alpha)
		ftod*= flt
		fft.ifft(ftod, model, normalize=True)
		del ftod
		# And subtract this from the actual data
		tod -= model

def broaden_beam_hor(tod, scan, ibeam, obeam):
	ft    = fft.rfft(tod)
	k     = 2*np.pi*fft.rfftfreq(scan.nsamp, 1/scan.srate)
	el    = np.mean(scan.box[:,2])
	skyspeed = scan.speed*np.cos(el)
	sigma = (obeam**2-ibeam**2)**0.5
	ft *= np.exp(-0.5*(sigma/skyspeed)**2*k**2)
	fft.ifft(ft, tod, normalize=True)

class FilterHighpass:
	def __init__(self, fknee=1.0, alpha=-3.5):
		"""Apply a highpass-filter to the TOD"""
		self.fknee = fknee
		self.alpha = alpha
	def __call__(self, scan, tod):
		f     = fft.rfftfreq(scan.nsamp, 1/scan.srate)
		ftod  = fft.rfft(tod)
		flt   = (1 + (np.maximum(f, f[1]/2)/self.fknee)**self.alpha)**-1
		ftod *= flt
		fft.ifft(ftod, tod, normalize=True)

class FilterBandpass:
	def __init__(self, fknee1=1.0, fknee2=2.0, alpha=-10):
		"""Apply a bandpass-filter to the TOD"""
		self.fknee1 = fknee1
		self.fknee2 = fknee2
		self.alpha  = alpha
	def __call__(self, scan, tod):
		f     = fft.rfftfreq(scan.nsamp, 1/scan.srate)
		f     = np.maximum(f, f[1]/2)
		flt1  = (1+(f/self.fknee1)**self.alpha)**-1 if self.fknee1 > 0 else f*0+1
		flt2  = (1+(f/self.fknee2)**self.alpha)**-1 if self.fknee2 > 0 else f*0
		flt   = flt1-flt2
		ftod  = fft.rfft(tod)
		ftod *= flt
		fft.ifft(ftod, tod, normalize=True)

class FilterBadify:
	def __init__(self, gainerr=0, deterr=0, deterr_rand=0, seed=0, tconsterr=0):
		"""Introduce systematic errors into the TOD. Only gain supported for now.
		Time constants should be easy. Pointing will be hard to support since div will
		already have been built by this point."""
		self.gainerr   = gainerr
		self.deterr    = deterr
		self.deterr_rand = deterr_rand
		self.tconsterr = tconsterr
		self.seed      = seed
	def __call__(self, scan, tod):
		# Build our random number generator
		import random
		seed2 = random.Random(scan.entry.id).randrange(0,0x10000000)
		rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence((self.seed, seed2))))
		# Gain errors
		det_muls  = 1 + rs.randn()*self.gainerr
		det_muls *= 1 + rs.randn(scan.ndet)*self.deterr_rand
		# random det errs. This didn't seem to do much
		# det_muls = [1+deterr*np.random.standard_normal(scan.ndet) for scan in myscans]
		# radial det errs
		det_r  = np.sum((scan.offsets[:,1:]-np.mean(scan.offsets[:,1:],0))**2,1)**0.5
		det_g  = utils.rescale(-det_r, [-self.deterr, self.deterr])
		det_g += 1 - np.mean(det_g)
		det_muls *= det_g
		# Then apply
		tod *= det_muls[:,None]
		# Maybe apply time constant error
		if self.tconsterr != 0:
			from enact import filters
			taus = rs.randn(scan.ndet)*self.tconsterr
			freq = fft.rfftfreq(scan.nsamp, 1/scan.srate)
			ft   = fft.rfft(tod)
			for di in range(ndet):
				ft[di] /= filters.tconst_filter(freq, taus[di])
			fft.irfft(ft, tod, normalize=True)

class FilterGainfit:
	"""Build a common mode and fit each detector to it, getting an estimate for
	the relative gain. Use these to recalibrate the tod in-place."""
	def __init__(self, fmax=1.0, odir=None):
		self.fmax = fmax
		self.odir = odir
	def __call__(self, scan, tod):
		# median to avoid being too affected by huge signals like bright point sources.
		# We want to be atmosphere-dominated
		common = np.median(tod,0)
		# Smooth away non-atmospheric stuff, which would dilute the fit
		freq   = fft.rfftfreq(scan.nsamp, 1/scan.srate)
		fmask  = (freq>0)&(freq<=self.fmax)
		fft.irfft(fft.rfft(common)*fmask, common, normalize=True)
		# Do the fit
		amps   = np.sum(tod*common,-1)/np.sum(common**2,-1)
		# And apply it
		tod   /= amps[:,None]
		# Optionally dump fits
		if self.odir:
			from enact import actdata
			utils.mkdir(self.odir)
			ofile = self.odir + "/cmodefit_%s.txt" % (scan.id.replace(":","_"))
			dets  = actdata.split_detname(scan.dets)[1]
			np.savetxt(ofile, np.concatenate([dets[None], scan.d.point_template.T/utils.degree, amps[None]]).T, fmt="%4d %8.5f %8.5f %8.5f")

class FilterCommon:
	"""Fit a common mode to the detectors, and either subtract it or replace the tod with it.
	This is the simplest case of high/low-pass filtering the focalplane"""
	def __init__(self, mode="highpass"):
		self.mode = mode
	def __call__(self, scan, tod):
		# median to avoid being too affected by single detectors
		common = np.median(tod,0)
		# We could smooth common in time too, if we suspect time-dependent gain errors
		# Not implemented for now
		if self.mode == "highpass":
			tod -= common
		elif self.mode == "lowpass":
			tod[:] = common
		else:
			raise ValueError("Unrecognized FilterCommon mode '%s'" % (str(self.mode)))

class FilterReplaceGain:
	def __call__(self, scan, tod):
		"""Test suggested by Thibaut. Use one set of gains to build the noise model,
		and another set to calibrate the data. Implemented by running this filter
		after the noise model has been built"""
		tod *= (scan.d.gain2 / scan.d.gain)[:,None]

###### Map filters ######

class MapfilterGauss:
	def __init__(self, scale, cap=None):
		self.scale = scale
		self.cap   = cap
	def __call__(self, map):
		l2 = np.sum(map.lmap()**2,0)
		f  = np.exp(-0.5*l2*self.scale**2)
		if self.scale < 0: f = 1-f
		if self.cap:
			f = np.maximum(f, 1/self.cap)
		f = f.astype(map.dtype)
		res = enmap.harm2map(enmap.map2harm(map)/f)
		res = enmap.samewcs(np.ascontiguousarray(res), res)
		return res

###### Composite operations #######

class SourceHandler:
	def __init__(self, scans, comm, srcs=None, tol=100, amplim=None, dtype=np.float64, rel=False, mode="full", inpainter="constrained", hits=True):
		for scan in scans:
			# Compute the source cut
			if srcs is None: srcs = scan.pointsrcs
			srcparam = pointsrcs.src2param(srcs).astype(np.float64)
			if amplim is not None:
				srcparam = srcparam[srcparam[:,2]>amplim]
			if rel: srcparam[:,2:5] /= srcparam[:,2,None]
			psrc     = pmat.PmatPtsrc(scan, srcparam)
			tod      = np.zeros((scan.ndet,scan.nsamp), np.float32)
			psrc.forward(tod, srcparam)
			mask     = tod > tol; del tod
			src_cut  = sampcut.from_mask(mask); del mask
			scan.cut_noiseest *= src_cut
			scan.src_cut       = src_cut
		self.scans     = scans
		self.mode      = mode
		self.inpainter = inpainter
		self.calc_hits = hits
		self.comm      = comm
		self.dtype     = dtype
		# signals
		self.signals   = []
		self.src_rhs   = []
		self.work      = []
		self.saved     = None
	def add_signal(self, signal):
		self.signals.append(signal)
		self.src_rhs.append(signal.zeros())
		self.work.append(signal.work())
	def filter(self, scan, tod): pass
	def filter2(self, scan, tod):
		if self.mode == "full":
			self.saved = scan.src_cut.extract_samples(tod)
		if self.inpainter == "joneig":
			gapfill.gapfill_joneig(tod, scan.src_cut, inplace=True)
		elif self.inpainter == "constrained":
			gapfill.gapfill_constrained(tod, scan.src_cut, scan.noise, maxiter=40, inplace=True)
		else: raise ValueError("Unrecognized inpainting method '%s'" % self.inpainter)
		if self.mode != "full": return
		# Make white noise maps of the sources
		src_tod  = tod.copy()
		scan.src_cut.insert_samples(src_tod, self.saved)
		src_tod -= tod
		sampcut.gapfill_const(scan.cut * ~scan.src_cut, src_tod, 0, inplace=True)
		# src_tod now contains a atm+cmb-cleaned version of the point source samples,
		# and is zero outside the source cut. Accumulate it into a src_rhs per
		# signal
		scan.noise.white(src_tod)
		with bench.mark("srch_rhs_PT"):
			for signal, rhs, work in zip(self.signals, self.src_rhs, self.work):
				# Project onto signals
				with bench.mark("srch_rhs_PT_" + signal.name):
					signal.precompute(scan)
					signal.backward(scan, src_tod, work)
					signal.free()
		del src_tod
		times = [bench.stats[s]["time"].last for s in ["srch_rhs_PT"]]
		L.debug("srch rhs P' %5.3f %s" % (tuple(times)+(scan.id,)))
	def post_b(self):
		if self.mode != "full": return
		# Finalize rhs
		with bench.mark("srch_rhs_reduce"):
			for signal, rhs, work in zip(self.signals, self.src_rhs, self.work):
				signal.finish(rhs, work)
		# To compute the div map we need a SignalCut for the source-only samples.
		# Getting this was surpisingly hacky :(
		cutlist = [scan.cut * ~scan.src_cut for scan in self.scans]
		# We can now build the src_divs
		self.src_divs = []
		self.src_hits = []
		self.src_maps = []
		for si, signal in enumerate(self.signals):
			div = signal.zeros(mat=True)
			calc_div_map(div, signal, self.scans, weights=[], cuts=cutlist, noise=True)
			if self.calc_hits:
				# Build hitcount map too
				hits = signal.zeros()
				hits = calc_hits_map(hits, signal, self.scans, cuts=cutlist)
			else: hits = None
			idiv = signal.polinv(div)
			map  = signal.polmul(idiv, self.src_rhs[si])
			self.src_divs.append(div)
			self.src_maps.append(map)
			self.src_hits.append(hits)
		self.src_rhs = None
	def write(self, prefix):
		if self.mode != "full": return
		for si, signal in enumerate(self.signals):
			signal.write(prefix, "srcmap", self.src_maps[si])
			signal.write(prefix, "srcdiv", self.src_divs[si])
			if self.src_hits[si] is not None:
				signal.write(prefix, "srchits", self.src_hits[si])

######## Equation system ########

class Eqsys:
	def __init__(self, scans, signals, filters=[], filters2=[], weights=[], multiposts=[],
			dtype=np.float64, comm=None, filters_noisebuild=[], white_noise=False):
		self.scans   = scans
		self.signals = signals
		self.dtype   = dtype
		self.filters = filters
		self.filters2= filters2
		self.filters_noisebuild = filters_noisebuild
		self.multiposts= multiposts
		self.weights = weights
		self.dof     = zipper.MultiZipper([signal.dof for signal in signals], comm=comm)
		self.b       = None
		self.white   = white_noise
	def A(self, x, debug_file=None):
		"""Apply the A-matrix P'N"P to the zipped vector x, returning the result."""
		with bench.mark("A_init"):
			imaps  = self.dof.unzip(x)
			omaps  = [signal.zeros() for signal in self.signals]
			# Set up our input and output work arrays. The output work array will accumulate
			# the results, so it must start at zero.
			iwork = [signal.filter(signal.prepare(map)) for signal, map in zip(self.signals, imaps)]
			owork = [signal.work() for signal in self.signals]
			#owork = [signal.prepare(map) for signal, map in zip(self.signals, omaps)]
		for si, scan in enumerate(self.scans):
			# Set up a TOD for this scan
			tod = np.zeros([scan.ndet, scan.nsamp], self.dtype)
			# Project each signal onto the TOD (P) in reverse order. This is done
			# so that the cuts can override the other signals.
			with bench.mark("A_P"):
				for signal, work in list(zip(self.signals, iwork))[::-1]:
					with bench.mark("A_Pr_" + signal.name):
						signal.precompute(scan)
					with bench.mark("A_P_" + signal.name):
						signal.forward(scan, tod, work)
			if debug_file is not None and si == 0:
				print("Eqsys A dumping debug")
				with h5py.File(debug_file,"w") as hfile:
					hfile["tod"]  = tod[:16]
					hfile["mask"] = scan.cut[:16].to_mask()
					hfile["dets"] = scan.dets[:16]
					hfile["id"]   = scan.id
			# Apply the noise matrix (N")
			with bench.mark("A_N"):
				for weight in self.weights: weight(scan, tod)
				if self.white: scan.noise.white(tod)
				else:          scan.noise.apply(tod)
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
				work = signal.filter(work)
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
	def calc_b(self, tod_provider=None):
		"""Compute b = P'N"d, and store it as the .b member. This involves
		reading in the TOD data and potentially estimating a noise model,
		so it is a heavy operation."""
		maps  = [signal.zeros() for signal in self.signals]
		owork = [signal.work() for signal in self.signals]
		#owork = [signal.prepare(map) for signal, map in zip(self.signals,maps)]
		for scan in self.scans:
			# Get the actual TOD samples (d)
			if tod_provider is None:
				with bench.mark("b_read"):
					tod  = scan.get_samples(verbose=False)
					tod  = tod.astype(self.dtype)
					tod  = utils.deslope(tod)
			else:
				tod = tod_provider(scan)
			#dump("dump_getsamples.hdf", tod)
			#dump("dump_prefilter_mean.hdf", np.mean(tod,0))
			#dump("dump_prefilter.hdf", tod[:4])
			#dump("hwp.hdf", scan.hwp)
			#1/0
			# Apply all filters (pickup filter, src subtraction, etc)
			if not config.get("debug_raw"):
				with bench.mark("b_filter"):
					for filter in self.filters: filter(scan, tod)
			#dump("dump_postfilter.hdf", tod)
			#dump("dump_postfilter_mean.hdf", np.mean(tod,0))
			#dump("dump_postfilter.hdf", tod[:4])
			#1/0
			# Apply the noise model (N")
			with bench.mark("b_weight"):
				for weight in self.weights: weight(scan, tod)
			#dump("dump_postweight.hdf", tod)
			#dump("dump_prenoise.hdf", tod[:32])
			with bench.mark("b_N_build"):
				if self.filters_noisebuild:
					tod_orig = tod.copy()
					for filter in self.filters_noisebuild: filter(scan, tod)
					tod = utils.deslope(tod)
				scan.noise = scan.noise.update(tod, scan.srate)
				if self.filters_noisebuild:
					tod = tod_orig
					del tod_orig
			#dump("dump_postupdate.hdf", tod)
			with bench.mark("b_filter2"):
				for filter in self.filters2: filter(scan, tod)
			#dump("dump_prenoise.hdf", tod)
			with bench.mark("b_N"):
				if self.white: scan.noise.white(tod)
				else:          scan.noise.apply(tod)
			#dump("dump_postnoise.hdf", tod)
			#1/0
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
				work = signal.filter(work)
				signal.finish(map, work)
		with bench.mark("b_zip"):
			self.b = self.dof.zip(maps)
	def dot(self, a, b):
		with bench.mark("dot"):
			return self.dof.dot(a,b)
	def postprocess(self, x):
		maps = self.dof.unzip(x)
		for multipost in self.multiposts:
			multipost(self.signals, maps)
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
				print("----", np.sum(ovec), ind)
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

class Eqsys2:
	# Like Eqsys, but takes a single Signals-signal instead of a list of signals
	def __init__(self, scans, signals, filters=[], filters2=[], weights=[], multiposts=[],
			dtype=np.float64, comm=None, filters_noisebuild=[], white_noise=False):
		self.scans   = scans
		self.signals = signals
		self.dtype   = dtype
		self.filters = filters
		self.filters2= filters2
		self.filters_noisebuild = filters_noisebuild
		self.multiposts= multiposts
		self.weights = weights
		self.dof     = signals.dof
		self.b       = None
		self.white   = white_noise
	def A(self, x, debug_file=None):
		"""Apply the A-matrix P'N"P to the zipped vector x, returning the result."""
		with bench.mark("A_init"):
			imaps  = self.dof.unzip(x)
			omaps  = self.signals.zeros()
			# Set up our input and output work arrays. The output work array will accumulate
			# the results, so it must start at zero.
			iwork = self.signals.filter(self.signals.prepare(imaps))
			owork = self.signals.work()
		for si, scan in enumerate(self.scans):
			# Set up a TOD for this scan
			tod = np.zeros([scan.ndet, scan.nsamp], self.dtype)
			# Project each signal onto the TOD (P) in reverse order. This is done
			# so that the cuts can override the other signals.
			with bench.mark("A_P"):
				self.signals.precompute(scan)
				self.signals.forward(scan, tod, iwork)
			# Apply the noise matrix (N")
			with bench.mark("A_N"):
				for weight in self.weights: weight(scan, tod)
				if self.white: scan.noise.white(tod)
				else:          scan.noise.apply(tod)
				for weight in self.weights[::-1]: weight(scan, tod)
			# Project the TOD onto each signal (P') in normal order. This is done
			# to allow the cuts to zero out the relevant TOD samples first
			with bench.mark("A_PT"):
				self.signals.backward(scan, tod, owork)
				self.signals.free()
			times = [bench.stats[s]["time"].last for s in ["A_P","A_N","A_PT"]]
			L.debug("A P %5.3f N %5.3f P' %5.3f %s %4d" % (tuple(times)+(scan.id,scan.ndet)))
		del iwork
		# Collect all the results, and flatten them
		with bench.mark("A_reduce"):
			# Looks like this could use more memory than the original
			owork = self.signals.filter(owork)
			self.signals.finish(omaps, owork)
		# priors
		with bench.mark("A_prior"):
			self.signals.prior(self.scans, imaps, omaps)
		return self.dof.zip(omaps)
	def M(self, x):
		"""Apply the preconditioner to the zipped vector x."""
		with bench.mark("M"):
			maps = self.dof.unzip(x)
			self.signals.precon(maps)
			return self.dof.zip(maps)
	def calc_b(self, tod_provider=None):
		"""Compute b = P'N"d, and store it as the .b member. This involves
		reading in the TOD data and potentially estimating a noise model,
		so it is a heavy operation."""
		maps = self.signals.zeros()
		work = self.signals.work()
		for scan in self.scans:
			# Get the actual TOD samples (d)
			if tod_provider is None:
				with bench.mark("b_read"):
					tod  = scan.get_samples(verbose=False)
					tod  = tod.astype(self.dtype)
					tod  = utils.deslope(tod)
			else:
				tod = tod_provider(scan)
			# Apply all filters (pickup filter, src subtraction, etc)
			if not config.get("debug_raw"):
				with bench.mark("b_filter"):
					for filter in self.filters: filter(scan, tod)
			# Apply the noise model (N")
			with bench.mark("b_weight"):
				for weight in self.weights: weight(scan, tod)
			with bench.mark("b_N_build"):
				if self.filters_noisebuild:
					tod_orig = tod.copy()
					for filter in self.filters_noisebuild: filter(scan, tod)
					tod = utils.deslope(tod)
				scan.noise = scan.noise.update(tod, scan.srate)
				if self.filters_noisebuild:
					tod = tod_orig
					del tod_orig
			with bench.mark("b_filter2"):
				for filter in self.filters2: filter(scan, tod)
			with bench.mark("b_N"):
				if self.white: scan.noise.white(tod)
				else:          scan.noise.apply(tod)
			with bench.mark("b_weight"):
				for weight in self.weights[::-1]: weight(scan, tod)
			# Project onto signals
			with bench.mark("b_PT"):
				self.signals.precompute(scan)
				self.signals.backward(scan, tod, work)
				self.signals.free()
			del tod
			times = [bench.stats[s]["time"].last for s in ["b_read","b_filter","b_N_build", "b_N", "b_PT"]]
			L.debug("b get %5.1f f %5.1f NB %5.3f N %5.3f P' %5.3f %s" % (tuple(times)+(scan.id,)))
		# Collect results
		with bench.mark("b_reduce"):
			self.signals.filter(work)
			self.signals.finish(maps, work)
		with bench.mark("b_zip"):
			self.b = self.dof.zip(maps)
	def dot(self, a, b):
		with bench.mark("dot"):
			return self.dof.dot(a,b)
	def postprocess(self, x):
		maps = self.dof.unzip(x)
		# This could use more memory than the previous version
		maps = self.signals.postprocess(maps)
		return self.dof.zip(maps)
	def write(self, prefix, tag, x):
		maps = self.dof.unzip(x)
		self.signals.write(prefix, tag, maps)

def write_precons(signals, prefix):
	for signal in signals:
		signal.precon.write(prefix)
