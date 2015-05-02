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
from enlib import enmap, utils

class Dmap:
	def __init__(self, shape, wcs, boxes, tshape=(240,240), dtype=None, comm=None):
		if comm is None: comm = mpi4py.MPI.COMM_WORLD
		shape = tuple(shape)
		tshape= tuple(tshape[-2:])
		pre   = tuple(shape[:-2])
		prelen= np.product(pre)
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
		wslices = gather(utils.box_slice(bbox, tbox),comm) # slices into work
		tslices = gather(utils.box_slice(tbox, bbox),comm) # slices into tiles
		overlaps    = utils.box_area(wslices)
		ownership   = assign_cols_round_robin(overlaps)
		print ownership
		# 5. Define tiles
		tiles = []
		for tb in tbox[ownership==comm.rank]:
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
			# Slices for transfering to and from w buffer
			for ws in wslices[comm.rank,ownership==id]:
				wlen = utils.box_area(ws)*prelen
				work_slice = (Ellipsis,slice(ws[0,0],ws[1,0]),slice(ws[0,1],ws[1,1]))
				wbuf_slice = slice(woff,woff+wlen)
				winfo.append((work_slice,wbuf_slice))
				woff += wlen
			for ti, ts in enumerate(tslices[id,ownership==comm.rank]):
				tlen = utils.box_area(ts)*prelen
				tile_slice = (Ellipsis,slice(ts[0,0],ts[1,0]),slice(ts[0,1],ts[1,1]))
				tbuf_slice = slice(toff,toff+tlen)
				tinfo.append((ti,tile_slice,tbuf_slice))
				toff += tlen
			wbufinfo[0,id] = woff-wbufinfo[1,id]
			tbufinfo[0,id] = toff-tbufinfo[1,id]
		wbufinfo, tbufinfo = tuple(wbufinfo), tuple(tbufinfo)
		# 7. Create mpi buffers
		self.wbuf = np.zeros(woff,dtype=dtype)
		self.tbuf = np.zeros(toff,dtype=dtype)
		# 8. Store necessary info
		self.dtype = work.dtype
		self.comm  = comm
		self.shape, self.wcs  = shape, wcs
		self.ibox,  self.bbox = ibox,  bbox
		self.tbox,  self.overlaps = tbox, overlaps
		self.wslices,  self.tslices = wslices, tslices
		self.wbufinfo, self.winfo   = wbufinfo, winfo
		self.tbufinfo, self.tinfo   = tbufinfo, tinfo
		self.work  = work
		self.tiles = tiles
	def work2tile(self):
		"""Project from local workspaces into the distributed tiles. Multiple workspaces
		may overlap with a single tile. The contribution from each workspace is summed."""
		for ws, bs in self.winfo:
			self.wbuf[bs] = self.work[ws].reshape(-1)
		self.comm.Alltoallv((self.wbuf, self.wbufinfo), (self.tbuf, self.tbufinfo))
		for tile in self.tiles: tile[...] = 0
		for ti, ts, bs in self.tinfo:
			self.tiles[ti][ts] += self.tbuf[bs].reshape(self.tiles[ti][ts].shape)
	def tile2work(self):
		"""Project from tiles into the local workspaces."""
		for ti, ts, bs in self.tinfo:
			self.tbuf[bs] = self.tiles[ti][ts].reshape(-1)
		self.comm.Alltoallv((self.tbuf, self.tbufinfo), (self.wbuf, self.wbufinfo))
		for ws, bs in self.winfo:
			self.work[ws] = self.wbuf[bs].reshape(self.work[ws].shape)

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
	res = np.zeros((comm.size,)+a.shape,dtype=a.dtype)
	comm.Allgather(a, res)
	return res
