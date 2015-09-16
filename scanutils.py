import numpy as np
from enlib import scan as enscan, errors, utils, coordinates, dmap2 as dmap

def calc_sky_bbox_scan(scan, osys):
	icorners = utils.box2corners(scan.box)
	ocorners = np.array([coordinates.transform(scan.sys,osys,b[1:,None],time=scan.mjd0+b[0,None]/3600/24,site=scan.site)[::-1,0] for b in icorners])
	# Take care of angle wrapping along the ra direction
	ocorners[...,1] = utils.rewind(ocorners[...,1], ref="auto")
	return utils.bounding_box(ocorners)

def distribute_scans(myinds, mycosts, myboxes, comm):
	"""Given the costs[nmyscan] and bounding boxes[nmyscan,2,2] of our local scans,
	compute a new scan distribution that distributes costs
	relatively evenly while keeping all scans a task owns
	in a local area of the sky. Returns the new myinds[nmyscan2] (indices into
	global scans array), mysubs[nmyscan2] (workspace index for each of my new scans),
	mybbox [nmyscan2,2,2] (bounding boxes for the new scans).

	This function does not move the scan data to the new processes. This is
	up tot he caller."""
	all_costs = comm.allreduce(mycosts)
	all_boxes = np.array(comm.allreduce(myboxes))
	# Avoid angle wraps.
	all_boxes[...,1] = np.sort(utils.rewind(all_boxes[...,1], ref="auto"),-1)
	all_inds  = comm.allreduce(myinds)
	myinds_old = myinds
	# Split into nearby scans
	mygroups = dmap.split_boxes_rimwise(all_boxes, all_costs, comm.size)[comm.rank]
	myinds = [all_inds[i] for group in mygroups for i in group]
	mysubs = [gi for gi, group in enumerate(mygroups) for i in group]
	mybbox = [utils.bounding_box([all_boxes[i] for i in group]) for group in mygroups]
	return myinds, mysubs, mybbox

def get_scan_bounds(myscans):
	return np.array([[np.min(scan.boresight[:,2:0:-1],0),np.max(scan.boresight[:,2:0:-1],0)] for scan in myscans])

def classify_scanning_patterns(myscans, tol=0.5*utils.degree, comm=None):
	"""Classify scans into scanning patterns based on [az,el] bounds.
	Returns patterns[:,{ftom,to},{el,az}] and pids[len(myscans)], where
	pids contains the index of each myscan into patterns."""
	boxes = get_scan_bounds(myscans)
	rank  = np.full(len(boxes),comm.rank)
	if comm is not None:
		boxes = utils.allgatherv(boxes, comm)
		ranks = utils.allgatherv(rank,  comm)
	pids = utils.label_unique(boxes, axes=(1,2), atol=tol)
	npattern = np.max(pids)+1
	# For each scanning pattern, define a bounding box
	pboxes = np.array([utils.bounding_box(boxes[pids==pid]) for pid in xrange(npattern)])
	# Get the ids for the scans that we have
	if comm is not None:
		pids = pids[rank==comm.rank]
	return pboxes, pids
