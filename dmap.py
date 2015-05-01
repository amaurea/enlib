"""This module implements mpi-distributed maps, where what is conceptually
a single contiguous enmap is split into tiles, with with tile belonging to
an mpi task.

The map is assumed to correspond to data from a set of scans,
which are represented here as a box per scan, with the scans also belonging
to mpi tasks. It is in general impossible to ensure a one-to-one mapping
between scan groups and tile groups, so the scans a given task owns will not
completely overlap with the tiles it owns. But hopefully the overlap won't be
too bad.

The area covered by the bounding box of all the scans a task owns is its
workspace. Projecting from scans to tiles is in principle:

	for scan in scans: scan -> workspace
	for wi in winfo:
		wtile = workspace[wi.slice]
		send(wi.id, wtile)
	for ti in tinfo:
		wtile = recv(ti.id)
		tiles[ti.i][ti.slice] += wtile

And projecting back:

	for ti in tinfo:
		wtile = tiles[ti.i][ti.slice]
		send(ti.id, wtile)
	for wi in winfo:
		wtile = recv(wi.id)
		workspace[wi.slice] = wtile
	for scan in scans: workspace -> scan

The complication is that sends and receives have to happen at the same time.
The easiest way to do this is via alltoallv, which requires the use of
flattened arrays.
"""
import numpy as np, mpi4py.MPI

class Dmap:
	def __init__(self, shape, wcs, boxes, tshape=(240.240), dtype=None, comm=None):
		if comm is None: comm = mpi4py.MPI.COMM_WORLD
		shape = tuple(shape)
		tshape= tuple(tshape[-2:])
		# 0. Translate boxes to pixels
		boxes = np.asarray(boxes)
		ibox  = box2pix(shape, wcs, boxes)
		# 1. Compute local box
		bbox  = utils.bounding_box(ibox)
		# 2. Set up local workspace
		wshape, wwcs = enmap.slice_wcs(shape[-2:], wcs, (slice(bbox[0,0],bbox[1,0]),slice(bbox[0,1],bbox[1,1])))
		work  = enmap.zeros(shape[:-2]+wshape, wwcs, dtype=dtype)
		# 3. Define tiling. Each tile has shape tshape, starting from the (0,0) corner
		#    of the full map. Tiles at the edge are clipped, as pixels beyond the edge
		#    of the full map may have undefined wcs positions.
		tbox  = build_tiles(shape, tshape).reshape(-1,2,2)
		ntile = len(tbox)
		# 4. Define tile ownership. For each task compute the overlap of each tile with its
		#    workspace. Tiles are given in rounds to ensure as equal memory use as possible.
		#    Each round the task with the highest overlap clams its highest overlap tile,
		#    and so on.
		my_overlaps = utils.box_overlap(tbox, bbox)
		overlaps = np.zeros([comm.size,ntile],dtype=int)
		comm.Allgather(my_overlaps, overlaps)
		ownership = assign_cols_round_robin(overlap)
		# 5. Define tile-worspace send slices and receive slices
		# 6. Build alltoallv structures:
		#    a) workspace,winfo <-> data,id_lens
		#    b) mytiles,tinfo <-> data,id_lens
		# 7. Implement alltoallv call

def box2pix(shape, wcs, box):
	"""Convert one or several bounding boxes of shape [2,2] or [n,2,2]
	into pixel counding boxes in standard python half-open format."""
	box  = np.asarray(box)
	fbox = box.reshape(-1,2,2)
	# Must rollaxis because sky2pix expects [{dec,ra},...]
	ibox = enmap.sky2pix(shape, wcs, np.rollaxis(fbox,1), corner=True)
	ibox = np.array([np.floor(ibox[0]),np.ceil(ibox[1])]).astype(int)
	ibox = np.rollaxis(ibox, 1)
	return ibox.reshape(box)

def build_tiles(shape, tshape):
	"""Given a bounding shape and the target shape of each tile, returns
	an [ty,tx,{from,to},{y,x}] array containing the bounds of each tile."""
	sa, ta = np.array(shape[-2:]), np.array(tshape)
	ntile = (sa+ta-1)/ta
	tbox  = np.zeros(tuple(ntile)+(2,2),dtype=int)
	y = np.maximum(shape[0],np.arange(ntile[0]+1)*tshape[0])
	x = np.maximum(shape[1],np.arange(ntile[1]+1)*tshape[1])
	tbox[:,:,0,0] = y[:-1,None]
	tbox[:,:,1,0] = y[ 1:,None]
	tbox[:,:,0,1] = x[None,:-1]
	tbox[:,:,1,1] = x[None, 1:]
	return tbox

def assign_cols_round_robin(scores):
	"""Given a 2d array of scores[n,m], associate each column to a row in
	a round-robin fashion. The result is an ownership array of length [n],
	indicating which column owns which row. The assignment is done greedily,
	such that each column is given to the row with the highest score in
	their intersecting cell. But once a row has been given a column, it
	can't get a new one until everybody else have had their turn."""
	nr, nc = scores.shape
	ownership = np.zeros([nc],dtype=int)
	cmask = np.full(nc, True).astype(bool)
	ris, cis = np.arange(nr), np.arange(nc)
	while True:
		rmask = np.full(nr, True).astype(bool)
		for i in xrange(nr):
			free = (np.sum(rmask), np.sum(cmask))
			if free[1] == 0: break
			rl, cl = np.unravel_index(np.argmax(scores[rmask][:,cmask]),free)
			ri, ci = ris[rmask][rl], cis[cmask][cl]
			ownership[ci] = ri
			rmask[ri] = False
			cmask[ci] = False
		if free[1] == 0: break
	return ownership
