from __future__ import division, print_function
import numpy as np, logging, h5py, sys
from . import scan as enscan, errors, utils, coordinates, dmap
from enact import actdata, filedb
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
		except IOError:
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

def get_tod_groups(ids, samelen=True):
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
