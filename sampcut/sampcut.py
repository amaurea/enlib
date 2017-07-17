import numpy as np
from . import fortran_32, fortran_64

icore = fortran_32.fortran
def get_core(dtype):
	if dtype == np.float32: return fortran_32.fortran
	else:                   return fortran_64.fortran

class Sampcut:
	# Constructors
	def __init__(self, ranges, detmap, nsamp, copy=True):
		"""Construct a Sampcut given a flattened sample range
		list ranges[nrange,{from,to}] and a detector mapping
		detmap[ndet+1] where index i contains the index into
		ranges for the first sample of detector i. nsamp is
		the logical max size of the array the cuts correspond to
		(which is needed for operations like inverting the cuts)."""
		self.ranges = np.array(ranges, dtype=np.int32, copy=copy)
		self.detmap = np.array(detmap, dtype=np.int32, copy=copy)
		self.nsamp  = int(nsamp)
	@staticmethod
	def empty(ndet, nsamp):
		"""Create Sampcut for ndet detectors and nsamp samples, with
		no samples cut."""
		ranges = np.zeros((0,2),np.int32)
		detmap = np.zeros(ndet+1,np.int32)
		return Sampcut(ranges, detmap, nsamp, copy=False)
	@staticmethod
	def full(ndet, nsamp)
		"""Create Sampcut for ndet detectors and nsamp samples, with
		no samples cut."""
		return ~Sampcut.empty(ndet, nsamp)
	@staticmethod
	def from_list(rlist, nsamp):
		"""Construct a Sampcut given a list of ranges[ndet][:,{from,to}]."""
		if len(rlist) == 0:
			# No detectors present
			ranges = np.zeros([0,2],np.int32)
			detmap = np.zeros(1,np.int32)
		else:
			ranges = []
			detmap = np.zeros(len(rlist)+1,np.int32)
			for di, dlist in enumerate(rlist):
				if len(dlist) > 0:
					ranges.append(dlist)
					detmap[di+1] = detmap[di] + len(dlist)
			ranges = np.concatenate(ranges,0).astype(np.int32)
			if ranges.ndim != 2:
				raise ValueError("Can only construct Sampcut from list of format [det][cuts][2]")
		return Sampcut(ranges, detmap, nsamp)
	def to_list(self):
		"""Return a list of [ndet][ncut,2]"""
		return [self.ranges[self.detmap[i]:self.detmap[i+1]] for i in range(self.ndet)]
	@staticmethod
	def from_mask(mask):
		mask = np.asarray(mask)
		if mask.ndim == 1: mask = mask[None]
		assert mask.ndim == 2, "Sampcut.from_mask requires a 1 or 2 dimensional array, but got %d" % mask.ndim
		mask = mask.view(np.int8)
		ncut = icore.count_mask(mask.T)
		ranges = np.empty([ncut,2],np.int32)
		detmap = np.empty([mask.shape[0]+1],np.int32)
		icore.mask_to_cut(mask.T, ranges.T, detmap)
		return Sampcut(ranges, detmap, mask.shape[1])
	def to_mask(self, omask=None):
		if omask is None: omask = np.empty([self.ndet, self.nsamp], np.bool)
		omask = omask.view(np.int8)
		icore.cut_to_mask(self.ranges.T, self.detmap, omask.T)
		return omask.view(np.bool)
	@property
	def ndet(self): return len(self.detmap)-1
	@property
	def nrange(self): return self.detmap[-1]
	def sum(self, axis=None):
		"""Compute the number of cut samples. If axis == 1, then
		the result will be the per-detector sum. Otherwise it will
		be the total sum across all detectors."""
		ncut = np.zeros(self.ndet, np.int32)
		icore.cut_nsamp(self.ranges.T, self.detmap, ncut)
		if axis == 1: return ncut
		else: return np.sum(ncut)
	def copy(self): return Sampcut(self.ranges, self.detmap, self.nsamp)
	def repeat(self, n):
		"""Multiply the number of detectors by n. The new detectors will
		be copies of the existing ones. Mostly useful for broadcasting."""
		odetmap = np.zeros(self.ndet*n+1,np.int32)
		oranges = np.zeros([len(self.ranges)*n,2],np.int32)
		icore.cut_mul(self.ranges.T, self.detmap, n, oranges.T, odetmap)
		return Sampcut(oranges, odetmap, self.nsamp)
	def widen(self, n):
		# Allow us to pass either a single number or a separate pre and post padding
		n = np.zeros(2,np.int32)+n
		# Widen all the ranges
		ranges = self.ranges.copy()
		ranges[:,0] -= n[0]
		ranges[:,1] += n[1]
		# make sure we don't exceed our bounds
		np.clip(ranges, 0, self.nsamp, ranges)
		# merge any overlapping ranges
		odetmap, oranges = self.detmap.copy(), ranges.copy()
		icore.cut_union(ranges.T, self.detmap, oranges.T, odetmap)
		oranges = oranges[:odetmap[-1]]
		return Sampcut(oranges, odetmap, self.nsamp)
	def extract_samples(self, tod):
		return extract_samples(self, tod)
	def insert_samples(self, tod, samples):
		return insert_samples(self, tod, samples)
	def __len__(self): return self.ndet
	def __mul__(self, other):
		"""Compute the composition of these cuts and the right-hand-side,
		returning a new Sampcut that cuts anything either of them cut."""
		# Broadcast if necessary
		if self.ndet == 1 and other.ndet > 1:
			self  = self.repeat(other.ndet)
		elif other.ndet == 1 and self.ndet > 1:
			other = other.repeat(self.ndet)
		# Merge into a single set of cuts
		wdetmap = self.detmap.copy()
		wranges = np.zeros([self.ranges.shape[0]+other.ranges.shape[0],2],np.int32)
		icore.cut_stack(self.ranges.T, self.detmap, other.ranges.T, other.detmap,
				wranges.T, wdetmap)
		# And merge overlapping cuts into non-overlapping, bigger cuts
		odetmap, oranges = wdetmap.copy(), wranges.copy()
		icore.cut_union(wranges.T, wdetmap, oranges.T, odetmap)
		oranges = oranges[:odetmap[-1]]
		return Sampcut(oranges, odetmap, self.nsamp, copy=False)
	#def __add__(self, other):
	#	"""cut1 + cut2 stacks these cuts in the detector direction"""
	#	return stack(self, other)
	def __invert__(self):
		"""Make cut samples uncut, and vice versa"""
		oranges = np.zeros((len(self.ranges)+self.ndet,2),np.int32)
		odetmap = self.detmap.copy()
		icore.cut_invert(self.ranges.T, self.detmap, self.nsamp, oranges.T, odetmap)
		oranges = oranges[:odetmap[-1]]
		return Sampcut(oranges, odetmap, self.nsamp, copy=False)
	def __getitem__(self, sel):
		"""Extract a subset of detectors and/or samples. Always functions
		as a slice, so the reslut is a new Sampcut. Only standard slicing is
		allowed in the sample direction - no direct indexing or indexing by lists."""
		if not isinstance(sel, tuple): sel = (sel,)
		if len(sel) == 0: return self
		if len(sel) > 2: raise IndexError("Too many indices for Sampcut. At most 2 indices supported")
		# Handle detector part
		detinds = np.arange(self.ndet)[sel[0]]
		if isinstance(detinds, int): detinds = np.full(1,detinds,np.int32)
		res     = self.copy()
		icore.cut_detslice(self.ranges.T, self.detmap.T, detinds, res.ranges.T, res.detmap)
		res.detmap = res.detmap[:len(detinds)+1]
		res.ranges = res.ranges[:res.detmap[-1]]
		if len(sel) == 1: return res
		# Handle sample part
		sampslice = [sel[1].start, sel[1].stop, sel[1].step]
		if sampslice[2] is None: sampslice[2] = 1
		if sampslice[0] is None:
			sampslice[0] = 0 if sampslice[2] > 0 else self.nsamp-1
		if sampslice[1] is None:
			sampslice[1] = self.nsamp if sampslice[2] > 0 else -1
		# Make sure we don't exceed our bounds
		if sampslice[2] > 0:
			sampslice[0] = max(sampslice[0],0)
			sampslice[1] = min(sampslice[1],self.nsamp)
		else:
			sampslice[0] = min(sampslice[0],self.nsamp-1)
			sampslice[1] = max(sampslice[1],-1)
		res2 = res.copy()
		icore.cut_sampslice(res.ranges.T, res.detmap, sampslice, res2.ranges.T, res2.detmap)
		res2.ranges = res2.ranges[:res2.detmap[-1]]
		# Total number of samples also changes
		res2.nsamp = (sampslice[1]-sampslice[0]+sampslice[2]-np.sign(sampslice[2]))/sampslice[2]
		return res2
	def __str__(self): return "Sampcut(ndet:%d,nsamp:%d,ncut:%d,cfrac:%.1f%%)" % (
			self.ndet, self.nsamp, self.detmap[-1], 100.0*self.sum()/(self.ndet*self.nsamp))
	def __repr__(self): return "Sampcut(ranges=%s, detmap=%s, nsamp=%d)" % (
			str(self.ranges), str(self.detmap), self.nsamp)

def sampcut(ranges, detmap, nsamp, copy=True):
	"""Construct a new sampcut. Convenience wrapper for Sampcut"""
	return Sampcut(ranges, detmap, nsamp, copy=copy)
def empty(ndet, nsamp):
	"""Construct a Sampcut with ndet detectors and nsamp samples, with no samples cut"""
	return Sampcut.empty(ndet, nsamp)
def full(ndet, nsamp):
	"""Constrct a Sampcut with ndet detectors and nsamp samples where everything is cut"""
	return Sampcut.full(ndet, nsamp)
def from_list(rlist, nsamp):
	"""Construct a Sampcut from the given list [ndet][nrange][2]"""
	return Sampcut.from_list(rlist, nsamp)
def from_mask(mask):
	"""Construct a Sampcut from the given bool mask[ndet,nsamp]"""
	return Sampcut.from_mask(mask)

def stack(cuts):
	"""stack((c1, c2, ...)). Concatenates the sample cuts c1, c2, etc.
	from the provided list in the detector direction. For example,
	c3 = stack((c1,c2)). If c1 had 10 dets and c2 had 100 dets, the result
	will have 110 dets, starting with c1's dets. All cuts must agree on nsamp."""
	ranges = np.concatenate([c.ranges for c in cuts]).astype(np.int32, copy=False)
	detmap = [[0]]
	for c in cuts: detmap.append(c.detmap[1:]+detmap[-1][-1])
	detmap = np.concatenate(detmap).astype(np.int32, copy=False)
	return Sampcut(ranges, detmap, cuts[0].nsamp, copy=False)

def extract_samples(cut, tod):
	"""Copy out the samples indicated by the Sampcut cut from the given tod,
	and return them as a 1d array"""
	samples = np.empty(cut.sum(), tod.dtype)
	get_core(tod.dtype).cut_extract(cut.ranges.T, cut.detmap, tod.T, samples)
	return samples
def insert_samples(cut, tod, samples):
	"""Inverse of extract_samples. Inserts samples into tod at the location
	given by Sampcut cut"""
	get_core(tod.dtype).cut_insert(cut.ranges.T, cut.detmap, tod.T, samples)

def gapfill_const(cut, tod, value, inplace=False):
	"""Fill cut values in tod by the given value. Returns the result."""
	if not inplace: tod = tod.copy()
	get_core(tod.dtype).gapfill_const(cut.ranges.T, cut.detmap, tod.T, value)
	return tod
def gapfill_linear(cut, tod, context=1, inplace=False):
	"""Fill cut ranges in tod with straight lines. context determines
	how many samples at each edge of the cut to use for determining the
	start and end value for each straight line. Defaults to 1 sample.
	Returns the result."""
	if not inplace: tod = tod.copy()
	get_core(tod.dtype).gapfill_linear(cut.ranges.T, cut.detmap, tod.T, context)
	return tod
