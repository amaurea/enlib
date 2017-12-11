import numpy as np, h5py
from enlib import rangelist, sampcut, utils

# Flagranges implement the new cuts format,
# where samples are tagged with flags instead
# of just being on/off. They can then be queried
# with a given flag combination to get the actual
# boolean cuts
#
# I would like the in-memory format to represent
# all the fields of these cuts, and I want things
# to be simpler than the multirange/rangelist
# stuff from before.
#
# How about [flags,...][n,{from,to}] using
# array(object) and array(int)? If so, we need
# 1. expand
# 2. combine
# 3. flatten
#
# Boolean flags are handled with the same format -
# in this case the flags array has length 1 and
# name "bool".
#
# Storing this expanded format may seem expensive,
# but the collapsed format for N flags is only smaller
# than N times the collapsed format for 1 flag if the
# flags usually change together.
#
# The cuts format contains det_uid, which I think
# is too specific for this format. But if I don't
# store it I won't be able to read in data and
# write it out unmodified. Could put I/O in enact.
#
# Our constructor takes:
# 1. flag_vals[nflag,...][n,{from,to}]
# 2. flag_names[nflag] optional
# 3. derived_masks[nder,{norm,invert},
#
# Evaluating derived masks is very simple in the compressed format -
# just or and and them together. In the expanded format it is harder
# because the data points aren't at the same location.
#
# Perhaps I shoudl store the flattened version instead? This would
# make extration of single flags simple.

class Flagrange:
	def __init__(self, nsamp, index_stack, flag_stack, stack_bounds, dets=None,
			flag_names=None, derived_masks=None, derived_names=None,
			sample_offset=0):
		"""Construct a Flagrange.
		nsamp int is the total number of samples in the dataset the flagrange applies to.
		stack_bounds int[ndet+1] is the index of the boundaries between each det's ranges
		index_stack"""
		self.nsamp        = int(nsamp)
		self.sample_offset= int(sample_offset)
		self.index_stack  = np.array(index_stack, np.uint32)
		self.flag_stack   = np.array(flag_stack,  np.uint8)
		self.stack_bounds = np.array(stack_bounds,np.uint32)
		self.dets         = np.arange(self.stack_bounds.size-1, dtype=np.uint32)
		if dets is not None: self.dets = np.array(dets, np.uint32)
		self.flag_names   = ["flag%d" % i for i in range(self.nbyte*8)]
		if flag_names is not None: self.flag_names = list(flag_names)
		self.derived_masks= np.zeros((0,2,self.nbyte),np.uint8)
		if derived_masks is not None: self.derived_masks = np.array(derived_masks, np.uint8)
		self.derived_names = ["derived%d" % i for i in range(self.derived_masks.shape[0])]
		if derived_names is not None: self.derived_names = list(derived_names)
	@property
	def nbyte(self): return self.flag_stack.shape[-1]
	@property
	def nflag(self): return len(self.flag_names)
	@property
	def ndet(self): return len(self.dets)
	def copy(self): return Flagrange(
			self.nsamp, self.index_stack, self.flag_stack, self.stack_bounds,
			self.dets, self.flag_names, self.derived_masks, self.derived_names,
			self.sample_offset)
	def select(self, flags):
		"""Return a new flagrange where only the given flags are set.
		The other flags will still exist, but their bits will be zero."""
		if isinstance(flags, basestring): flags = [flags]
		# Build bitfield
		pos = np.zeros(self.nbyte, np.uint8)
		neg = pos.copy()
		for flag in flags:
			inverse = flag.startswith("~") or flag.startswith("!")
			if inverse: flag = flag[1:]
			try:
				i    = self.flag_names.index(flag)
				byte = i/8
				bit  = 1<<(i%8)
				if not inverse: pos[byte] |= bit
				else:           neg[byte] |= bit
			except ValueError:
				i = self.derived_names.index(flag)
				if not inverse:
					pos |= self.derived_masks[i,0]
					neg |= self.derived_masks[i,1]
				else:
					neg |= self.derived_masks[i,0]
					pos |= self.derived_masks[i,1]
		res = self.copy()
		res.flag_stack = self.flag_stack & pos | ~self.flag_stack & neg
		return res
	def __getitem__(self, sel):
		dslice = None
		sslice = None
		if isinstance(sel, tuple):
			if len(sel) > 0: dslice = sel[0]
			if len(sel) > 1: sslice = sel[1]
		else: dslice = sel
		return self.restrict(dslice, sslice)
	def restrict(self, dslice=None, sslice=None):
		if dslice is None: dslice = slice(None)
		if sslice is None: sslice = slice(None)
		if isinstance(dslice,(int,long)): dslice = [dslice]
		if isinstance(sslice,(int,long)): sslice = [sslice]
		# Extract sample slice parameters and handle negative slicing
		start, stop, step = sslice.start, sslice.stop, sslice.step
		if step is None: step = 1
		reverse = step < 0
		if not reverse:
			if start is None: start = 0
			if stop   is None: stop   = self.nsamp
		else:
			if start is None: start = self.nsamp-1
			if stop   is None: stop   = -1
			(start, stop, step) = (stop+1, start+1, -step)

		dets = self.dets[dslice]
		stot = 0
		flag_stack   = []
		index_stack  = []
		stack_bounds = []
		for di in np.arange(self.ndet)[dslice]:
			stack_bounds.append(stot)
			s1, s2 = self.stack_bounds[di:di+2]
			flags  = self.flag_stack  [s1:s2]
			inds   = self.index_stack [s1:s2]
			# Perform the sample slicing
			if reverse:
				inds  = self.nsamp-1-inds[::-1]
				flags = flags[::-1]
			good  = (inds>=start)&(inds<stop)
			inds  = inds[good]
			flags = flags[good]
			if len(inds) == 0: continue
			inds -= start
			inds /= step
			# Merge ones that fall on the same index
			oinds, oflags = [inds[0]], [flags[0]]
			for ind, flag in zip(inds, flags):
				if ind == oinds[-1]:
					oflags[-1] |= flag
				else:
					oinds.append(ind)
					oflags.append(flag)
			flag_stack.append(np.array(oflags))
			index_stack.append(np.array(oinds))
			stot += len(oinds)
		stack_bounds.append(stot)
		res = self.copy()
		res.flag_stack    = np.concatenate(flag_stack)
		res.index_stack   = np.concatenate(index_stack)
		res.stack_bounds  = np.array(stack_bounds)
		res.nsamp         = (stop-start)/step
		# This won't work when reversing, but then
		# sample offset doesn't make any sense for reversed arrays
		res.sample_offset = self.sample_offset + start
		res.dets = dets
		return res
	def count_flag_ranges(self, perdet=False):
		counts = np.zeros([self.ndet, self.nflag],int)
		for di in range(self.ndet):
			s1,s2 = self.stack_bounds[di:di+2]
			for fi in range(self.nflag):
				byte = fi/8
				bit  = 1<<(fi%8)
				counts[di,fi] = np.sum(self.flag_stack[s1:s2,byte]&bit>0)
		if not perdet: counts = np.sum(counts,0)
		return counts
	def count_flag_samples(self, perdet=False):
		counts = np.zeros([self.ndet, self.nflag],int)
		for di in range(self.ndet):
			s1,s2 = self.stack_bounds[di:di+2]
			inds  = self.index_stack[s1:s2]
			lens  = np.concatenate([inds[1:]-inds[:-1],[self.nsamp-inds[-1]]])
			for fi in range(self.nflag):
				byte = fi/8
				bit  = 1<<(fi%8)
				mask = self.flag_stack[s1:s2,byte]&bit>0
				counts[di,fi] = np.sum(lens*mask)
		if not perdet: counts = np.sum(counts,0)
		return counts
	def __repr__(self):
		fields = ["nsamp=%d" % self.nsamp, "ndet=%d" % self.ndet,
				"nrange=%d" % len(self.index_stack)]
		if len(self.flag_names) > 0:
			fields.append("flags_names=[%s]" % ",".join(self.flag_names))
		if len(self.derived_names) > 0:
			fields.append("derived_names=[%s]" % ",".join(self.derived_names))
		return "Flagrange(%s)" % ",".join(fields)
	def to_ranges(self):
		"""Return a list of ranges of nonzero flags"""
		ranges = []
		for di in range(self.ndet):
			s1, s2 = self.stack_bounds[di:di+2]
			inds   = self.index_stack[s1:s2]
			mask   = np.any(self.flag_stack[s1:s2],1)
			mask   = np.concatenate([[False],mask]).astype(int)
			edges  = mask[1:]-mask[:-1]
			starts = np.where(edges>0)[0]
			ends   = np.where(edges<0)[0]
			if len(starts) == 0:
				ranges.append(np.zeros([0,2],int))
			else:
				r = np.zeros([len(starts),2],int)
				r[:,0] = inds[starts]
				r[:len(ends),1] = inds[ends]
				if len(ends) < len(starts): r[-1,1] = self.nsamp
				ranges.append(r)
		return ranges
	def to_rangelist(self):
		ranges = self.to_ranges()
		return rangelist.Multirange([rangelist.Rangelist(r, n=self.nsamp) for r in ranges])
	def to_sampcut(self):
		# This could be optimized, since sampcuts and flagranges have some similariteis
		# int he internal representation
		ranges = self.to_ranges()
		return sampcut.from_list(ranges, self.nsamp)
	@staticmethod
	def from_sampcut(scut, dets=None, name="cut", sample_offset=0):
		from_sampcut(scut, dets=dets, name=name, sample_offset=sample_offset)
	def write(self, hfile, group=None):
		write_flagrange(hfile, self, group=group)

def from_sampcut(scut, dets=None, name="cut", sample_offset=0):
	# To zeroeth order, flags simply become 1 when we enter a cut range and 0 when we exit
	flag_stack  = scut.ranges.copy()
	flag_stack[:] = [1,0]
	flag_stack  = flag_stack.reshape(-1)
	index_stack = scut.ranges.reshape(-1)
	# However, each detector starts out uncut by default, so we must insert an uncut
	# at all the beginnings
	stack_bounds= utils.cumsum(2*scut.nranges, True)
	flag_stack  = np.insert(flag_stack, stack_bounds[:-1], 0)
	index_stack = np.insert(index_stack,stack_bounds[:-1], 0)
	stack_bounds= utils.cumsum(2*scut.nranges+1, True)
	# At this point we may have some empty ranges. I think that's acceptable
	# Expand flag_stack to full dimensionality
	flag_stack = flag_stack[:,None]
	return Flagrange(scut.nsamp, index_stack, flag_stack,
			stack_bounds, dets=dets, flag_names=[name], derived_names=["cuts"],
			derived_masks=[[[1],[0]]], sample_offset=sample_offset)

#def combine_flagranges(franges):
#	offsets = [frange.sample_offset for frange in franges]
#	minoff  = np.min(offsets)

def read_flagrange(hfile, group=None):
	if isinstance(hfile, basestring):
		with h5py.File(hfile, "r") as f:
			if group is not None: f = f[group]
			return read_flagrange(f)
	nsamp = 1000000
	sample_offset = 0
	attrs = hfile.attrs
	if "sample_count"  in attrs: nsamp         = attrs["sample_count"]
	if "sample_offset" in attrs: sample_offset = attrs["sample_offset"]
	return Flagrange(
		nsamp,
		hfile["index_stack"].value,
		hfile["flag_stack"].value,
		hfile["stack_bounds"].value,
		dets = hfile["det_uid"].value,
		flag_names = hfile["flag_names"].value,
		derived_masks = hfile["derived_masks"].value,
		derived_names = hfile["derived_names"].value,
		sample_offset = sample_offset,
	)

def write_flagrange(hfile, frange, group=None):
	if isinstance(hfile, basestring):
		with h5py.File(hfile, "w") as f:
			write_flagrange(f, frange, group=group)
	elif group is not None:
		g = hfile.create_group(group)
		write_flagrange(g, frange)
	else:
		hfile["det_uid"]      = frange.dets
		hfile["flag_names"]   = frange.flag_names
		hfile["stack_bounds"] = frange.stack_bounds
		hfile["flag_stack"]   = frange.flag_stack
		hfile["index_stack"]  = frange.index_stack
		hfile["derived_names"]= frange.derived_names
		hfile["derived_masks"]= frange.derived_masks
		hfile.attrs["sample_offset"]= frange.sample_offset
		hfile.attrs["sample_count"] = frange.nsamp

# While I don't like rangelist and multirange, I'll use them
# for now instead of building a new boolean range type.
#
#class Maskrange:
#	"""This class is a simple boolean version of Flagrange. It
#	can be used to represent a single flag."""
#	def __init__(self, nsamp, range_stack, stack_bounds, dets=None, sample_offset=0):
#		self.nsamp         = int(nsamp)
#		self.sample_offset = int(sample_offset)
#		self.stack_bounds  = np.array(index_stack, np.uint32)
#		self.range_stack   = np.array(range_stack, np.uint32)
#		self.dets          = np.arange(self.stack_bounds.size, dtype=np.uint32)
#		if dets is not None: self.dets = np.array(dets, np.uint32)

