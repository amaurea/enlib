"""This module deals with a Scan variant for point source analysis.
It is not a subclass of Scan as it makes incompatible assumptions,
and things are stored more explicitly."""
import numpy as np, h5py

class SrcScan:
	def __init__(self, tod, point, phase, ranges, rangesets, offsets, ivars):
		self.tod     = tod
		self.point   = point
		self.phase   = phase
		self.ranges  = ranges
		self.rangesets=rangesets
		self.offsets = offsets
		self.ivars   = ivars
	def __str__(self): return "SrcScan(nsrc=%d,ndet=%d,nsamp=%d)" % (self.offsets.shape[0],self.offsets.shape[1],self.tod.size)
	def __getitem__(self, sel):
		if type(sel) != tuple:
			sel = (sel,)
		sel = sel + (slice(None),)*(2-len(sel))
		nsrc, ndet = self.offsets[:,:-1].shape
		srcs = np.arange(nsrc)[sel[0]]
		dets = np.arange(ndet)[sel[1]]
		return self.select(srcs, dets)
	def select(self, srcs, dets):
		"""Extract a new SrcScan for the specified srcs and dets,
		eliminating ranges that are no longer needed."""
		# 1. First slice offsets and rangesets
		rangesets = []
		offsets = np.zeros([len(srcs),len(dets)+1],dtype=np.int32)
		for si, src in enumerate(srcs):
			for di, det in enumerate(dets):
				o1,o2 = self.offsets[src,det:det+2]
				offsets[si,di] = len(rangesets)
				rangesets.append(self.rangesets[o1:o2])
		rangesets = np.concatenate(rangesets)
		# 2. Then determine which ranges are no longer used, and
		# a mappings between old and new ranges
		used = np.zeros(len(self.ranges),dtype=bool)
		used[rangesets] = True
		rmap  = np.nonzero(used)[0]
		irmap = np.zeros(len.self.ranges)
		irmap[rmap] = np.arange(len(rmap))
		# 3. Extract valid ranges and update rangesets
		ranges = self.ranges[rmap]
		rangesets = irmap[rangesets]
		# 4. Extract our actual samples while updating ranges
		n = np.sum(ranges[:,1]-ranges[:,0])
		tod   = self.tod[:n].copy()
		point = self.point[:n].copy()
		phase = self.phase[:n].copy()
		m = 0
		for ri in range(len(ranges)):
			i1,i2 = ranges[ri]
			o1,o2 = m,m+i2-i1
			tod  [o1:o2] = self.tod [i1:i2]
			point[o1:o2] = self.point[i1:i2]
			phase[o1:o2] = self.phsae[i1:i2]
			ranges[ri] = [o1,o2]
			m = o2
		return SrcScan(tod, point, phase, ranges, rangesets, offsets, self.ivars[dets])

def write_srcscan(fname, scan):
	with h5py.File(fname, "w") as hfile:
		for key in ["tod","point","phase","ranges","rangesets","offsets","ivars"]:
			hfile[key] = getattr(scan, key)

def read_srcscan(fname):
	args = {}
	with h5py.File(fname, "r") as hfile:
		for key in ["tod","point","phase","ranges","rangesets","offsets","ivars"]:
			args[key] = hfile[key].value
	return SrcScan(**args)
