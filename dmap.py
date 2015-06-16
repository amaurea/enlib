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
import numpy as np, mpi4py.MPI, copy, os, re, enlib.slice, logging
from enlib import enmap, utils, zipper
from astropy.wcs import WCS
L = logging.getLogger(__name__)

import h5py

class Dmap:
	"""Dmap - distributed enmap. After construction, its relevant members
	are:
		.work:  list of local workspace data. Each workspace is an enmap
		.tiles: list of tile data owned by this task. Each tile is an enmap.
		.work2tile(): sums contribution from all workspaces into tiles
		.tile2work(): projects from tiles to workspaces
		.shape, .wcs: geometry of distributed map
		.dtype: dtype of all maps."""
	def __init__(self, shape, wcs, bbpix=None, bbox=None, tshape=None, dtype=None, comm=None):
		"""Construct a distributed map structure for the geometry specified by
		shape and wcs.

		bbox indicates the bounds [{from,to},{lat,lon}] of the area of the map of
		interest to each mpi task, and will in general be different for each task.
		bbpix is the same as bbox, but expressed in units of pixels.
		It is allowed to pass a list of bounding boxes, in which case each mpi task
		will have multiple work spaces.

		tshape specifies the tiling scheme to use. The global geometry will be
		split into tiles of tshape dimensions (except at the edges, as no tile
		will extend beyond the edge of the full map), and tiles will be stored
		in a distributed fashion between mpi tasks based on the degree of overlap
		with their workspaces."""
		if comm is None: comm = mpi4py.MPI.COMM_WORLD
		if tshape is None: tshape = (240,240)
		shape = tuple(shape)
		tshape= tuple(tshape[-2:])
		pre   = tuple(shape[:-2])
		prelen= np.product(pre)
		# 1. Compute local box
		if bbpix is None:
			bbpix = box2pix(shape, wcs, bbox)
		# For slicing, we need the first bound to be lower than the second bound.
		# It might not be, as Ra and x have opposite ordering. So sort along the
		# from-to axis. bbpix has 3 dims [scan,fromto,radec]
		bbpix = bbpix.reshape((-1,)+bbpix.shape[-2:])
		bbpix = np.sort(bbpix, 1)
		# Some scans may extend partially beyond the end of our map. We must therefore
		# truncate the bounding box.
		bbpix[:,0,:] = np.maximum(bbpix[:,0,:],0)
		bbpix[:,1,:] = np.maximum(bbpix[:,1,:],shape[-2:])
		# 2. Set up local workspace(s)
		work  = []
		for b in bbpix:
			wshape, wwcs = enmap.slice_wcs(shape[-2:], wcs, (slice(b[0,0],b[1,0]),slice(b[0,1],b[1,1])))
			work.append(enmap.zeros(shape[:-2]+wshape, wwcs, dtype=dtype))
		# 3. Define global workspace ownership
		nwork = utils.allgather([len(bbpix)],comm)
		wown  = np.concatenate([np.full(n,i,dtype=int) for i,n in enumerate(nwork)])
		# 3. Define tiling. Each tile has shape tshape, starting from the (0,0) corner
		#    of the full map. Tiles at the edge are clipped, as pixels beyond the edge
		#    of the full map may have undefined wcs positions.
		tbox   = build_tiles(shape, tshape)
		bshape = tbox.shape[:2]
		tbox   = tbox.reshape(-1,2,2)
		ntile  = len(tbox)
		bcoord = np.array([np.arange(ntile)/bshape[1],np.arange(ntile)%bshape[1]]).T
		# 4. Define tile ownership.
		# a) For each task compute the overlap of each tile with its workspaces, and
		#    concatenate across tasks to form a [nworktot,ntile] array.
		wslices = utils.allgatherv(utils.box_slice(bbpix, tbox),comm, axis=0) # slices into work
		tslices = utils.allgatherv(utils.box_slice(tbox, bbpix),comm, axis=1) # slices into tiles
		# b) Compute the total overlap each mpi task has with each tile, and use this
		# to decide who should get which tiles
		overlaps    = utils.box_area(wslices)
		overlaps    = utils.sum_by_id(overlaps, wown, 0)
		town        = assign_cols_round_robin(overlaps)
		# Map tile indices from local to global and back
		tgmap = [[] for i in range(comm.size)]
		tlmap = np.zeros(ntile,dtype=int)
		for ti, id in enumerate(town):
			tlmap[ti] = len(tgmap[id]) # glob 2 loc
			tgmap[id].append(ti)       # loc  2 glob
		# 5. Define tiles
		tiles = []
		for tb in tbox[town==comm.rank]:
			_, twcs = enmap.slice_wcs(shape[-2:], wcs, (slice(tb[0,0],tb[1,0]),slice(tb[0,1],tb[1,1])))
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
		self.pre   = pre
		self.bbpix,  self.tbox      = bbpix, tbox
		self.shape, self.wcs        = shape, wcs
		self.wown,  self.town       = wown, town
		self.tlmap, self.tgmap      = tlmap, tgmap
		self.tshape,   self.bshape  = tshape,  bshape
		self.wslices,  self.tslices = wslices, tslices
		self.wbufinfo, self.winfo   = wbufinfo, winfo
		self.tbufinfo, self.tinfo   = tbufinfo, tinfo
		self.bcoord = bcoord
		self.ntile  = ntile
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
		for i, (wi, ws, bs) in enumerate(self.winfo):
			self.work[wi][ws] = self.wbuf[bs].reshape(self.work[wi][ws].shape)
	def copy(self):
		res = copy.deepcopy(self)
		res.comm = self.comm
		return res
	@property
	def ndim(self): return len(self.shape)
	@property
	def npix(self): return np.product(self.shape[-2:])
	def astype(self, dtype):
		if dtype == self.dtype: return self
		else:
			res = self.copy()
			res.tiles = [t.astype(dtype) for t in self.tiles]
			res.work  = [w.astype(dtype) for w in self.work]
			return res
	def fill(self, val):
		for t in self.tiles: t[:] = val
		for w in self.work:  w[:] = val
		return self
	def __getitem__(self, sel):
		# Split sel into normal and wcs parts.
		sel1, sel2 = enlib.slice.split_slice(sel, [self.ndim-2,2])
		if len(sel2) > 0:
			raise NotImplementedError("Pixel slicing of dmaps not implemented")
		res = self.copy()
		# Tiles, work and buffers are affected by slicing. All collapsed indices must
		# be divided by the change in overall size
		res.pre= np.zeros(self.pre)[sel].shape
		prelen = np.product(res.pre)
		oldlen = np.product(self.pre)
		res.shape = res.pre + self.shape[-2:]
		res.wbufinfo = tuple(np.array(res.wbufinfo)*prelen/oldlen)
		res.tbufinfo = tuple(np.array(res.tbufinfo)*prelen/oldlen)
		res.wbuf = res.wbuf[:res.wbuf.size*prelen/oldlen]
		res.tbuf = res.tbuf[:res.tbuf.size*prelen/oldlen]
		res.winfo = [(a[0],a[1],slice(a[2].start*prelen/oldlen,a[2].stop*prelen/oldlen)) for a in self.winfo]
		res.tinfo = [(a[0],a[1],slice(a[2].start*prelen/oldlen,a[2].stop*prelen/oldlen)) for a in self.tinfo]
		for ti, oldtile in enumerate(self.tiles): res.tiles[ti] = oldtile[sel]
		for wi, oldwork in enumerate(self.work):  res.work[wi]  = oldwork[sel]
		return res
	def __setitem__(self, sel, val):
		# Split sel into normal and wcs parts.
		sel1, sel2 = enlib.slice.split_slice(sel, [self.ndim-2,2])
		if len(sel2) > 0:
			raise NotImplementedError("Pixel slicing of dmaps not implemented")
		try:
			for tile, vtile in zip(self.tiles, val.tiles): tile[sel] = vtile
			for work, vwork in zip(self.work,  val.work):  work[sel] = vwork
		except AttributeError:
			for tile in self.tiles: tile[sel] = val
			for work in self.work:  work[sel] = val

def write_map(name, map, ext="fits", merged=True):
	if not merged:
		# Write as individual tiles in directory of the specified name
		utils.mkdir(name)
		for id, tile in zip(map.tgmap[map.comm.rank],map.tiles):
			coords = map.bcoord[id]
			enmap.write_map(name + "/tile%03d_%03d.%s" % (tuple(coords)+(ext,)), tile)
	else:
		# Write to a single file. This currently creates the full map
		# in memory while writing. It is unclear how to avoid this
		# without bypassing pyfits or becoming super-slow.
		if map.comm.rank == 0:
			canvas = enmap.zeros(map.shape, map.wcs, map.dtype)
		else:
			canvas = None
		dmap2enmap(map, canvas)
		if map.comm.rank == 0:
			enmap.write_map(name, canvas)

def read_map(name, bbpix=None, bbox=None, tshape=None, comm=None):
	if comm is None: comm = mpi4py.MPI.COMM_WORLD
	if os.path.isdir(name):
		# Find the number of tiles in the map
		entries = os.listdir(name)
		nrow, ncol = 0,0
		for entry in entries:
			match = re.search(r'^tile(\d+)_(\d+).([^.]+)$', entry)
			if match:
				nrow = max(nrow,1+int(match.group(1)))
				ncol = max(ncol,1+int(match.group(2)))
				ext  = match.group(3)
		# Build the list of tile files
		tfiles = [["" for c in range(ncol)] for r in range(nrow)]
		for entry in entries:
			match = re.search(r'^tile(\d+)_(\d+).([^.]+)$', entry)
			if match: tfiles[int(match.group(1))][int(match.group(2))] = entry
		if nrow == 0: raise IOError("'%s' is not a valid dmap file" % name)
		# Find the tile size and map extent
		tile1 = enmap.read_map(name+"/"+tfiles[0][0])
		tile2 = enmap.read_map(name+"/"+tfiles[-1][-1])
		npre, tshape = tile1.shape[:-2], tile1.shape[-2:]
		wcs = tile1.wcs
		shape = npre + (tile1.shape[-2]*(nrow-1)+tile2.shape[-2],tile1.shape[-1]*(ncol-1)+tile2.shape[-1])
		dtype = tile1.dtype
		# Construct our dmap and read our tiles
		map = Dmap(shape, wcs, bbpix=bbpix, bbox=bbox, tshape=tshape, dtype=dtype, comm=comm)
		for id, tile in zip(map.tgmap[map.comm.rank],map.tiles):
			coords = map.bcoord[id]
			tile[:] = enmap.read_map(name+"/"+tfiles[coords[0]][coords[1]])
	else:
		# Map is in a single file. Get map info
		if comm.rank == 0:
			canvas = enmap.read_map(name)
			shape = comm.bcast(canvas.shape)
			wcs   = WCS(comm.bcast(canvas.wcs.to_header_string()))
			dtype = comm.bcast(canvas.dtype)
			# Hack: Pickling changes the dtype from < to =, both of which are
			# equivalent. But mpi has problems with the former.
			canvas.dtype = dtype
		else:
			shape = comm.bcast(None)
			wcs   = WCS(comm.bcast(None))
			dtype = comm.bcast(None)
			canvas= None
		map = Dmap(shape, wcs, bbpix=bbpix, bbox=bbox, tshape=tshape, dtype=dtype, comm=comm)
		# And send data to the tiles
		enmap2dmap(canvas, map)
	return map

def enmap2dmap(emap, dmap, root=0):
	"""Import data from an enmap into a dmap."""
	for ti in range(dmap.ntile):
		id  = dmap.town[ti]
		loc = dmap.tlmap[ti]
		box = dmap.tbox[ti]
		if dmap.comm.rank == root:
			data = np.ascontiguousarray(emap[...,box[0,0]:box[1,0],box[0,1]:box[1,1]])
		if dmap.comm.rank == root and id == root:
			dmap.tiles[loc] = data
		elif dmap.comm.rank == root:
			dmap.comm.Send(data, dest=id, tag=loc)
		elif dmap.comm.rank == id:
			dmap.comm.Recv(dmap.tiles[loc], source=root, tag=loc)

def dmap2enmap(dmap, emap, root=0):
	"""Transfer data from a a dmap to a full enmap hosted on
	the mpi task with id given by root."""
	for ti in range(dmap.ntile):
		id  = dmap.town[ti]
		loc = dmap.tlmap[ti]
		box = dmap.tbox[ti]
		if dmap.comm.rank == root and id == root:
			data = dmap.tiles[loc]
		elif dmap.comm.rank == root:
			data = np.zeros(dmap.pre+tuple(box[1]-box[0]), dtype=dmap.dtype)
			dmap.comm.Recv(data, source=id, tag=loc)
		elif dmap.comm.rank == id:
			dmap.comm.Send(dmap.tiles[loc], dest=root, tag=loc)
		if dmap.comm.rank == root:
			emap[...,box[0,0]:box[1,0],box[0,1]:box[1,1]] = data

def box2pix(shape, wcs, box):
	"""Convert one or several bounding boxes of shape [2,2] or [n,2,2]
	into pixel counding boxes in standard python half-open format."""
	box  = np.asarray(box)
	fbox = box.reshape(-1,2,2)
	# Must rollaxis because sky2pix expects [{dec,ra},...]
	ibox = enmap.sky2pix(shape, wcs, utils.moveaxis(fbox,2,0), corner=True)
	ibox = np.array([np.floor(ibox[0]),np.ceil(ibox[1])]).astype(int)
	ibox = utils.moveaxis(ibox, 0, 2)
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

def split_boxes_rimwise(boxes, weights, nsplit):
	"""Given a list of bounding boxes[nbox,{from,to},ndim] and
	an array of weights[nbox], compute how to split the boxes
	into nsplit subsets such that the total weights for each
	split is reasonably even while the bounding box for each
	group is relatively small. Returns nsplit lists of
	indices into boxes, where each sublist should be treated
	given a separate bounding box (though this function
	always returns only a single list per split).

	The algorithm used is a greedy one which repeatedly picks
	the box futherst from the center and builds a group based
	on its nearest point. For a box distribution without
	large holes in it, this should result in a somewhat even
	distribution, but it is definitely not optimal.
	"""
	weights = np.asarray(weights)
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

class DmapZipper(zipper.ArrayZipper):
	"""Zips and unzips Dmap objects. Only the tile data is
	zipped. A Dmap is always assumed to be distributed, so there is no
	"shared" argument."""
	def __init__(self, template, mask=None):
		zipper.SingleZipper.__init__(self, False, template.comm)
		self.template, self.mask = template, mask
		if self.mask is None:
			cum = utils.cumsum([t.size for t in self.template.tiles], endpoint=True)
		else:
			cum = utils.cumsum([np.sum(m) for m in self.mask.tiles], endpoint=True)
		self.n = cum[-1]
		self.bins = np.array([cum[:-1],cum[1:]]).T
	def zip(self, a):
		if self.mask is None:
			return np.concatenate([t.reshape(-1) for t in a.tiles])
		else:
			return np.concatenate([t[m] for t,m in zip(a.tiles,self.mask.tiles)])
	def unzip(self, x):
		if self.mask is None:
			for b,t in zip(self.bins, self.template.tiles):
				t[...] = x[b[0]:b[1]].reshape(t.shape)
		else:
			for b,t,m in zip(self.bins, self.template.tiles, self.mask.tiles):
				t[m] = x[b[0]:b[1]]
		return self.template
