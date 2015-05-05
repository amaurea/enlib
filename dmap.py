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
import numpy as np, mpi4py.MPI, copy
from enlib import enmap, utils

class Dmap:
	"""Dmap - distributed enmap. After construction, its relevant members
	are:
		.work:  list of local workspace data. Each workspace is an enmap
		.tiles: list of tile data owned by this task. Each tile is an enmap.
		.work2tile(): sums contribution from all workspaces into tiles
		.tile2work(): projects from tiles to workspaces
		.shape, .wcs: geometry of distributed map
		.dtype: dtype of all maps."""
	def __init__(self, shape, wcs, bbox, tshape=(240,240), dtype=None, comm=None):
		"""Construct a distributed map structure for the geometry specified by
		shape and wcs.

		bbox indicates the bounds [{from,to},{lat,lon}] of the area of the map of
		interest to each mpi task, and will in general be different for each task.
		It is allowed to pass a list of bounding boxes, in which case each mpi task
		will have multiple work spaces.

		tshape specifies the tiling scheme to use. The global geometry will be
		split into tiles of tshape dimensions (except at the edges, as no tile
		will extend beyond the edge of the full map), and tiles will be stored
		in a distributed fashion between mpi tasks based on the degree of overlap
		with their workspaces."""
		if comm is None: comm = mpi4py.MPI.COMM_WORLD
		shape = tuple(shape)
		tshape= tuple(tshape[-2:])
		pre   = tuple(shape[:-2])
		prelen= np.product(pre)
		# 1. Compute local box
		bbox  = box2pix(shape, wcs, bbox)
		bbox  = bbox.reshape((-1,)+bbox.shape[-2:])
		# 2. Set up local workspace(s)
		work  = []
		for b in bbox:
			wshape, wwcs = enmap.slice_wcs(shape[-2:], wcs, (slice(b[0,0],b[1,0]),slice(b[0,1],b[1,1])))
			work.append(enmap.zeros(shape[:-2]+wshape, wwcs, dtype=dtype))
		# 3. Define global workspace ownership
		nwork = gather([len(bbox)],comm)
		wown  = np.concatenate([np.full(n,i,dtype=int) for i,n in enumerate(nwork)])
		# 3. Define tiling. Each tile has shape tshape, starting from the (0,0) corner
		#    of the full map. Tiles at the edge are clipped, as pixels beyond the edge
		#    of the full map may have undefined wcs positions.
		tbox  = build_tiles(shape, tshape).reshape(-1,2,2)
		ntile = len(tbox)
		# 4. Define tile ownership.
		# a) For each task compute the overlap of each tile with its workspaces, and
		#    concatenate across tasks to form a [nworktot,ntile] array.
		wslices = gatherv(utils.box_slice(bbox, tbox),comm, axis=0) # slices into work
		tslices = gatherv(utils.box_slice(tbox, bbox),comm, axis=1) # slices into tiles
		# b) Compute the total overlap each mpi task has with each tile, and use this
		# to decide who should get which tiles
		overlaps    = utils.box_area(wslices)
		overlaps    = utils.sum_by_id(overlaps, wown, 0)
		town        = assign_cols_round_robin(overlaps)
		# 5. Define tiles
		tiles = []
		for tb in tbox[town==comm.rank]:
			tshape, twcs = enmap.slice_wcs(shape[-2:], wcs, (slice(tb[0,0],tb[1,0]),slice(tb[0,1],tb[1,1])))
			tiles.append(enmap.zeros(shape[:-2]+tuple(tb[1]-tb[0]), twcs, dtype=dtype))
		# 6. Define mapping between work<->wbuf and tiles<->tbuf
		wbufinfo  = np.zeros([2,comm.size],dtype=int)
		tbufinfo  = np.zeros([2,comm.size],dtype=int)
		winfo, tinfo = [], []
		woff, toff = 0, 0
		for id in xrange(comm.size):
			## Buffer info to send to alltoallv
			wbufinfo[1,id] = woff
			tbufinfo[1,id] = toff
			# Slices for transfering to and from w buffer. Loop over all of my
			# workspaces and determine the slices into them and how much we need
			# to send.
			for tloc, tglob in enumerate(np.where(town==id)[0]):
				for wloc, wglob in enumerate(np.where(wown==comm.rank)[0]):
					ws = wslices[wglob,tglob]
					wlen = utils.box_area(ws)*prelen
					work_slice = (Ellipsis,slice(ws[0,0],ws[1,0]),slice(ws[0,1],ws[1,1]))
					wbuf_slice = slice(woff,woff+wlen)
					winfo.append((wloc,work_slice,wbuf_slice))
					woff += wlen
			# Slices for transferring to and from t buffer. Loop over all
			# my tiles, and determine how much I have to receive from each
			# workspace of each task.
			for tloc, tglob in enumerate(np.where(town==comm.rank)[0]):
				for wloc, wglob in enumerate(np.where(wown==id)[0]):
					ts = tslices[tglob,wglob]
					tlen = utils.box_area(ts)*prelen
					tile_slice = (Ellipsis,slice(ts[0,0],ts[1,0]),slice(ts[0,1],ts[1,1]))
					tbuf_slice = slice(toff,toff+tlen)
					tinfo.append((tloc,tile_slice,tbuf_slice))
					toff += tlen
			wbufinfo[0,id] = woff-wbufinfo[1,id]
			tbufinfo[0,id] = toff-tbufinfo[1,id]
		wbufinfo, tbufinfo = tuple(wbufinfo), tuple(tbufinfo)
		# 7. Create mpi buffers
		self.wbuf = np.zeros(woff,dtype=dtype)
		self.tbuf = np.zeros(toff,dtype=dtype)
		# 8. Store necessary info
		self.dtype = np.dtype(dtype)
		self.comm  = comm
		self.bbox,  self.tbox       = bbox, tbox
		self.shape, self.wcs        = shape, wcs
		self.wown,  self.town       = wown, town
		self.wslices,  self.tslices = wslices, tslices
		self.wbufinfo, self.winfo   = wbufinfo, winfo
		self.tbufinfo, self.tinfo   = tbufinfo, tinfo
		self.work  = work
		self.tiles = tiles
	def work2tile(self):
		"""Project from local workspaces into the distributed tiles. Multiple workspaces
		may overlap with a single tile. The contribution from each workspace is summed."""
		for wi, ws, bs in self.winfo:
			self.wbuf[bs] = self.work[wi][ws].reshape(-1)
		self.comm.Alltoallv((self.wbuf, self.wbufinfo), (self.tbuf, self.tbufinfo))
		for tile in self.tiles: tile[...] = 0
		for ti, ts, bs in self.tinfo:
			self.tiles[ti][ts] += self.tbuf[bs].reshape(self.tiles[ti][ts].shape)
	def tile2work(self):
		"""Project from tiles into the local workspaces."""
		for ti, ts, bs in self.tinfo:
			self.tbuf[bs] = self.tiles[ti][ts].reshape(-1)
		self.comm.Alltoallv((self.tbuf, self.tbufinfo), (self.wbuf, self.wbufinfo))
		for wi, ws, bs in self.winfo:
			self.work[wi][ws] = self.wbuf[bs].reshape(self.work[wi][ws].shape)
	def copy(self):
		return copy.deepcopy(self)

def box2pix(shape, wcs, box):
	"""Convert one or several bounding boxes of shape [2,2] or [n,2,2]
	into pixel counding boxes in standard python half-open format."""
	box  = np.asarray(box)
	fbox = box.reshape(-1,2,2)
	# Must rollaxis because sky2pix expects [{dec,ra},...]
	ibox = enmap.sky2pix(shape, wcs, np.rollaxis(fbox,1), corner=True)
	ibox = np.array([np.floor(ibox[0]),np.ceil(ibox[1])]).astype(int)
	ibox = np.rollaxis(ibox, 1)
	return ibox.reshape(box.shape)

def build_tiles(shape, tshape):
	"""Given a bounding shape and the target shape of each tile, returns
	an [ty,tx,{from,to},{y,x}] array containing the bounds of each tile."""
	sa, ta = np.array(shape[-2:]), np.array(tshape)
	ntile = (sa+ta-1)/ta
	tbox  = np.zeros(tuple(ntile)+(2,2),dtype=int)
	y = np.minimum(sa[0],np.arange(ntile[0]+1)*ta[0])
	x = np.minimum(sa[1],np.arange(ntile[1]+1)*ta[1])
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

def gather(a, comm):
	a   = np.asarray(a)
	res = np.zeros((comm.size,)+a.shape,dtype=a.dtype)
	comm.Allgather(a, res)
	return res

def gatherv(a, comm, axis=0):
	"""Perform an mpi allgatherv along the specified axis of the array
	a, returning an array with the individual process arrays concatenated
	along that dimension. For example gatherv([[1,2]],comm) on one task
	and gatherv([[3,4],[5,6]],comm) on another task results in
	[[1,2],[3,4],[5,6]] for both tasks."""
	fa = utils.moveaxis(a, axis, 0)
	ra = fa.reshape(fa.shape[0],-1)
	N  = ra.shape[1]
	n  = gather([len(ra)],comm)
	o  = utils.cumsum(n)
	rb = np.zeros((np.sum(n),N),dtype=ra.dtype)
	comm.Allgatherv(ra, (rb, (n*N,o*N)))
	fb = rb.reshape((rb.shape[0],)+fa.shape[1:])
	return utils.moveaxis(fb, 0, axis)

def split_boxes_rimwise(boxes, weights, nsplit):
	"""Given a list of bounding boxes[nbox,{from,to},ndim] and
	an array of weights[nbox], compute how to split the boxes
	into nsplit subsets such that the total weights for each
	split is reasonably even while the bounding box for each
	group is relatively small. Returns nsplit lists lists of
	indices into boxes, where each sublist should be treated
	given a separate bounding box (though this function
	always returns only a single list per split).

	The algorithm used is a greedy one which repeatedly picks
	the box futherst from the center and builds a group based
	on its nearest point. For a box distribution without
	large holes in it, this should result in a somewhat even
	distribution, but it is definitely not optimal.
	"""
	# Divide boxes into N groups with as equal weight as possible,
	# and as small bbox as possible
	n = len(boxes)
	groups = []
	# Compute distance of every point from center. We will
	# start consuming points from edges
	centers    = np.mean(boxes,1)
	center_tot = np.mean(centers,0)
	cdist      = calc_dist2(centers, center_tot[None])
	totweight  = np.sum(weights)
	# We keep track of which boxes have already been
	# processed via a mask.
	mask = np.full(n, True, dtype=np.bool)
	cumweight  = 0
	for gi in xrange(nsplit):
		# Compute the target weight for this group.
		# On average this should simply be totweight/nsplit,
		# but we adjust it on the fly to compensate for any
		# groups that end up deviating from this.
		targweight = (totweight-cumweight)/(nsplit-gi)
		p = unmask(np.argmax(cdist[mask]),mask)
		mask[p] = False
		# Find distance of every point to this point. Ouch, this
		# makes the algorithm O(N^2) if one doesn't introduce gridding
		pdist = calc_dist2(centers[mask], centers[p,None])
		dinds = unmask(np.argsort(pdist),mask)
		cumw  = np.cumsum(weights[dinds])
		# We will use as many of the closest points as
		# needed to reach the target weight, but not
		# so many that there aren't enough points left
		# for at least one per remaining mpi task.
		if gi == nsplit-1:
			nsel = None
		else:
			nsel = len(np.where(cumw < targweight)[0])
			nsel = max(0,min(nsel, np.sum(mask)-(nsplit-gi)))
		group = np.concatenate([[p],dinds[:nsel]])
		groups.append([group])
		mask[group] = False
		cumweight += np.sum(weights[group])
	return groups

def calc_dist2(a,b): return np.sum((a-b)**2,1)
def unmask(inds, mask): return np.where(mask)[0][inds]
