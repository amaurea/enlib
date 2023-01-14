from __future__ import division, print_function
import numpy as np, logging, h5py, sys
from . import scan as enscan, errors, utils, coordinates, dmap, bunch
from enact import actdata, filedb
from scipy import ndimage
L = logging.getLogger(__name__)

try: basestring
except: basestring = str

def calc_sky_bbox_scan(scan, osys, nsamp=100):
	"""Compute the bounding box of the scan in the osys coordinate system.
	Returns [{from,to},{dec,ra}]."""
	ipoints = utils.box2contour(scan.box, nsamp)
	opoints = np.array([coordinates.transform(scan.sys,osys,b[1:,None],time=scan.mjd0+b[0,None]/3600/24,site=scan.site)[::-1,0] for b in ipoints])
	# Take care of angle wrapping along the ra direction
	opoints[...,1] = utils.rewind(opoints[...,1], ref="auto")
	obox = utils.bounding_box(opoints)
	# Grow slighly to account for non-infinite nsamp
	obox = utils.widen_box(obox, 5*utils.arcmin, relative=False)
	return obox

def distribute_scans(myinds, mycosts, myboxes, comm):
	"""Given the costs[nmyscan] and bounding boxes[nmyscan,2,2] of our local scans,
	compute a new scan distribution that distributes costs
	relatively evenly while keeping all scans a task owns
	in a local area of the sky. Returns the new myinds[nmyscan2] (indices into
	global scans array), mysubs[nmyscan2] (workspace index for each of my new scans),
	mybbox [nmyscan2,2,2] (bounding boxes for the new scans).

	This function does not move the scan data to the new processes. This is
	up tot he caller."""
	all_costs = np.array(comm.allreduce(mycosts))
	all_inds  = np.array(comm.allreduce(myinds))
	if myboxes is None:
		myinds = all_inds[utils.equal_split(all_costs, comm.size)[comm.rank]]
		return myinds
	else:
		all_boxes = np.array(comm.allreduce(myboxes))
		# Avoid angle wraps. We assume that the boxes are all correctly wrapped
		# individually.
		ras    = all_boxes[:,:,1]
		ra_ref = np.median(np.mean(ras,1))
		rashift= ra_ref + (ras[:,0]-ra_ref+np.pi)%(2*np.pi) - np.pi - ras[:,0]
		ras += rashift[:,None]
		all_boxes[:,:,1] = ras
		# Split into nearby scans
		mygroups = dmap.split_boxes_rimwise(all_boxes, all_costs, comm.size)[comm.rank]
		myinds = [all_inds[i] for group in mygroups for i in group]
		mysubs = [gi for gi, group in enumerate(mygroups) for i in group]
		mybbox = [utils.bounding_box([all_boxes[i] for i in group]) for group in mygroups]
		return myinds, mysubs, mybbox

def distribute_scans2(inds, costs, comm, boxes=None):
	"""Given the costs[nscan] and bounding boxes[nscan,2,2] of all scans,
	compute a new scan distribution that distributes costs
	relatively evenly while keeping all scans a task owns
	in a local area of the sky. Returns the new myinds[nmyscan] (indices into
	global scans array), mysubs[nmyscan] (workspace index for each of my new scans),
	mybbox[nmyscan,2,2] (bounding boxes for the new scans)."""
	if boxes is None:
		return inds[utils.equal_split(costs, comm.size)[comm.rank]]
	else:
		# Avoid angle wraps. We assume that the boxes are all correctly wrapped
		# individually.
		ras    = boxes[:,:,1]
		ra_ref = np.median(np.mean(ras,1))
		rashift= ra_ref + (ras[:,0]-ra_ref+np.pi)%(2*np.pi) - np.pi - ras[:,0]
		ras += rashift[:,None]
		boxes[:,:,1] = ras
		# Split into nearby scans
		mygroups = dmap.split_boxes_rimwise(boxes, costs, comm.size)[comm.rank]
		myinds = [inds[i] for group in mygroups for i in group]
		mysubs = [gi for gi, group in enumerate(mygroups) for i in group]
		mybbox = [utils.bounding_box([boxes[i] for i in group]) if len(group) > 0 else boxes[0] for group in mygroups]
		return myinds, mysubs, mybbox

def get_scan_bounds(myscans, ref=0):
	bounds = np.array([[np.min(scan.boresight[:,2:0:-1],0),np.max(scan.boresight[:,2:0:-1],0)] for scan in myscans])
	# Resolve az wrap, assuming no scans crossing straight north or south. We also make
	# this assumption in pmat_core. This could be generalized if necessary.
	bounds[...,1] = utils.rewind(bounds[...,1], ref=ref)
	return bounds

def classify_scanning_patterns(myscans, tol=0.5*utils.degree, comm=None):
	"""Classify scans into scanning patterns based on [az,el] bounds.
	Returns patterns[:,{ftom,to},{el,az}] and pids[len(myscans)], where
	pids contains the index of each myscan into patterns."""
	boxes = get_scan_bounds(myscans)
	rank  = np.full(len(boxes),comm.rank)
	if comm is not None:
		boxes = utils.allgatherv(boxes, comm)
		rank  = utils.allgatherv(rank,  comm)
	pids = utils.label_unique(boxes, axes=(1,2), atol=tol)
	npattern = np.max(pids)+1
	# For each scanning pattern, define a bounding box
	pboxes = np.array([utils.bounding_box(boxes[pids==pid]) for pid in range(npattern)])
	# Get the ids for the scans that we have
	if comm is not None:
		pids = pids[rank==comm.rank]
	return pboxes, pids

def scan_iterator(filelist, inds, reader, db=None, dets=None, quiet=False, downsample=1, hwp_resample=False):
	"""Given a set of ids/files and a set of indices into that list. Try
	to read each of these scans. Returns a list of successfully read scans
	and a list of their indices."""
	for ind in inds:
		try:
			if not isinstance(filelist[ind],basestring): raise IOError
			d = enscan.read_scan(filelist[ind])
			#actdata.read(filedb.data[filelist[ind]])
		except (IOError, OSError):
			try:
				entry = db[filelist[ind]]
				d = reader(entry)
				if d.ndet == 0 or d.nsamp == 0:
					raise errors.DataMissing("Tod contains no valid data")
			except errors.DataMissing as e:
				if not quiet: L.debug("Skipped %s (%s)" % (str(filelist[ind]), e.args[0]))
				continue
		if dets:
			if dets.startswith("@"):
				uids = [int(line.split()[0]) for line in open(dets[1:],"r")]
				_, duids = actdata.split_detname(d.dets)
				_,det_inds = utils.common_inds([uids,duids])
				d = d[det_inds]
			else:
				d = eval("d[%s]" % dets)
		hwp_active = np.any(d.hwp_phase[0] != 0)
		if hwp_resample and hwp_active:
			mapping = enscan.build_hwp_sample_mapping(d.hwp)
			d = d.resample(mapping)
		d = d[:,::downsample]
		if not quiet: L.debug("Read %s" % str(filelist[ind]))
		yield ind, d

def read_scans(filelist, inds, reader, db=None, dets=None, quiet=False, downsample=1, hwp_resample=False):
	"""Given a set of ids/files and a set of indices into that list. Try
	to read each of these scans. Returns a list of successfully read scans
	and a list of their indices."""
	myinds, myscans  = [], []
	for ind, scan in scan_iterator(filelist, inds, reader, db=db, dets=dets, quiet=quiet, downsample=downsample, hwp_resample=hwp_resample):
		myinds.append(ind)
		myscans.append(scan)
	return myinds, myscans

def read_scans_autobalance(ids, reader, comm, db=None, dets=None, quiet=False, downsample=1,
		hwp_resample=False, sky_local=False, osys="equ"):
	"""Read the given list of ids/files, and try to distribute them across the tasks of MPI communicator
	comm as balancedly as possible. if sky_local=True, then the distribution will also try to keep the
	tods belonging to each mpi task in a local area of the sky.

	Returns a Bunch with members:
	.n:     The total number of scans read. Check if this is zero to see if there's anything to do.
	.scans: The list of scans for this mpi task
	.inds:  The corresponding indices into the ids list
	.bbox:  The bounding box of this tasks's scans on the sky. Only computed if sky_local is True,
	        otherwise this is None.
	"""
	if not quiet: L.info("Reading %d scans" % len(ids))
	myinds = np.arange(len(ids))[comm.rank::comm.size]
	myinds, myscans = read_scans(ids, myinds, reader, db=db, dets=dets, downsample=downsample,
			hwp_resample=hwp_resample, quiet=quiet)
	myinds = np.array(myinds, int)

	# Collect scan info. This currently fails if any task has empty myinds
	read_ids  = [ids[ind] for ind in utils.allgatherv(myinds, comm)]
	read_ntot = len(read_ids)
	if not quiet: L.info("Found %d tods" % read_ntot)
	if read_ntot == 0: return bunch.Bunch(n=0, inds=[], scans=[], bbox=None, subs=None, autocuts=None)

	# Some scans may have been cut by the autocuts. Save that information here,
	# since it would be lost otherwise
	read_ndets= utils.allgatherv([len(scan.dets) for scan in myscans], comm)
	ncut = np.sum(read_ndets==0)
	nok  = np.sum(read_ndets >0)
	try: # Apparently this can fail
		autocuts = bunch.Bunch(
				names= [cut[0] for cut in myscans[0].autocut],
				cuts = utils.allgatherv(np.array([[cut[1:] for cut in scan.autocut] for scan in myscans]),comm),
				ids  = read_ids)
	except (AttributeError, IndexError):
		autocuts = None

	# Prune fully autocut scans, now that we have output the autocuts
	mydets  = [len(scan.dets) for scan in myscans]
	myinds  = [ind  for ind, ndet in zip(myinds, mydets) if ndet > 0]
	myscans = [scan for scan,ndet in zip(myscans,mydets) if ndet > 0]
	if not quiet: L.info("Pruned %d fully autocut tods" % ncut)

	# Try to get about the same amount of data for each mpi task.
	# If we use distributed maps, we also try to make things as local as possible
	mycosts = [s.nsamp*s.ndet for s in myscans]
	if sky_local: # distributed maps
		myboxes = [calc_sky_bbox_scan(s, osys) for s in myscans]
		myinds, mysubs, mybbox = distribute_scans(myinds, mycosts, myboxes, comm)
	else:
		myinds = distribute_scans(myinds, mycosts, None, comm)
		mybbox, mysubs = None, None
	del myscans # scans do take up some space, even without the tod being read in

	# And reread the correct files this time. Ideally we would
	# transfer this with an mpi all-to-all, but then we would
	# need to serialize and unserialize lots of data, which
	# would require lots of code.
	if not quiet: L.info("Rereading shuffled scans")
	myinds, myscans = read_scans(ids, myinds, reader, db=db, dets=dets, downsample=downsample,
			hwp_resample=hwp_resample, quiet=quiet)

	# Get the index into the global list of accepted scans
	allinds = utils.allgatherv(myinds, comm)
	rinds   = utils.find(allinds, myinds)

	return bunch.Bunch(n=nok, scans=myscans, inds=myinds, rinds=rinds, bbox=mybbox, subs=mysubs, autocuts=autocuts)

def get_tod_groups(ids, samelen=True):
	"""Given a set of ids. Return a list of groups of ids. Each croup consists of
	ids that cover the same time period with a different array.
	
	The function currently assumes that the tods must have *exactly* the same
	starting label, which is usually but not always the case.
	"""
	times = np.array([float(id[:id.index(".")]) for id in ids])
	labels = utils.label_unique(times, rtol=0, atol=10)
	nlabel = np.max(labels)+1
	groups = [ids[labels==label] for label in range(nlabel)]
	# Try to preserve original ordering
	first  = [group[0] for group in groups]
	orig_inds = utils.find(ids, first)
	order  = np.argsort(orig_inds)
	groups = [groups[i] for i in order]
	if samelen:
		nsub = np.max(np.bincount(labels))
		groups = [g for g in groups if len(g) == nsub]
	return groups

def find_scan_periods(db, ttol=60, atol=2*utils.degree, mindur=120):
	"""Given a scan db, return the set of contiguous scanning periods in the form
	[:,{ctime_from,ctime_to}]."""
	atol = atol/utils.degree
	info = np.array([filedb.scans.data[a] for a in ["baz", "bel", "waz", "wel", "t", "dur"]]).T
	# Get rid of nan entries
	bad  = np.any(~np.isfinite(info),1)
	# get rid of too short tods, since those don't have reliable az bounds
	bad |= info[:,-1] < mindur
	info = info[~bad]
	t1   = info[:,-2] - info[:,-1]/2
	info = info[np.argsort(t1)]
	# Start, end
	t1   = info[:,-2] - info[:,-1]/2
	t2   = t1 + info[:,-1]
	# Remove angle ambiguities
	info[:,0] = utils.rewind(info[:,0], period=360)
	# How to find jumps:
	# 1. It's a jump if the scanning changes
	# 2. It's also a jump if a the interval between tod-ends and tod-starts becomes too big
	changes    = np.abs(info[1:,:4]-info[:-1,:4])
	jumps      = np.any(changes > atol,1)
	jumps      = np.concatenate([[0], jumps]) # from diff-inds to normal inds
	# Time in the middle of each gap
	gap_times = np.mean(find_period_gaps(np.array([t1,t2]).T, ttol=ttol),1)
	gap_inds  = np.searchsorted(t1, gap_times)
	jumps[gap_inds] = True
	# raw:  aaaabbbbcccc
	# diff: 00010001000
	# 0pre: 000010001000
	# cum:  000011112222
	labels  = np.cumsum(jumps)
	linds   = np.arange(np.max(labels)+1)
	t1s     = ndimage.minimum(t1, labels, linds)
	t2s     = ndimage.maximum(t2, labels, linds)
	# Periods is [nperiod,{start,end}] in ctime. Start is the start of the first tod
	# in the scanning period. End is the end of the last tod in the scanning period.
	periods = np.array([t1s, t2s]).T
	return periods

def find_period_gaps(periods, ttol=60):
	"""Helper for find_scan_periods. Given the [:,{ctime_from,ctime_to}] for all
	the individual scans, returns the times at which the gap between the end of
	a tod and the start of the next is greater than ttol (default 60 seconds)."""
	# We want to sort these and look for any places
	# where a to is followed by a from too far away. To to this we need to keep
	# track of which entries in the combined, sorted array was a from or a to
	periods = np.asarray(periods)
	types   = np.zeros(periods.shape, int)
	types[:,1] = 1
	types   = types.reshape(-1)
	ts      = periods.reshape(-1)
	order   = np.argsort(ts)
	ts, types = ts[order], types[order]
	# Now look for jumps
	jumps = np.where((ts[1:]-ts[:-1] > ttol) & (types[1:]-types[:-1] < 0))[0]
	# We will return the time corresponding to each gap
	gap_times = np.array([ts[jumps], ts[jumps+1]]).T
	return gap_times
