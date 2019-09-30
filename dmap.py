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
import numpy as np, copy, os, re, operator
from . import enmap, utils, zipper, mpi
from astropy.wcs import WCS

def empty(geometry):
	"""Return an empty Dmap with the specified geometry."""
	return Dmap(geometry)
def zeros(geometry):
	"""Return a new Dmap with the specified geometry, filled with zeros."""
	tiles = [enmap.zeros(ts,tw,dtype=geometry.dtype) for ts,tw in geometry.loc_geometry]
	return Dmap(geometry, tiles, copy=False)
def ones(geometry):
	"""Return a new Dmap with the specified geometry, filled with ones."""
	tiles = [enmap.ones(ts,tw,dtype=geometry.dtype) for ts,tw in geometry.loc_geometry]
	return Dmap(geometry, tiles, copy=False)
def full(geometry, val):
	"""Return a new Dmap fwith the specified geometry, filled with the given value."""
	tiles = [enmap.full(ts,tw,val,dtype=geometry.dtype) for ts,tw in geometry.loc_geometry]
	return Dmap(geometry, tiles, copy=False)

def geometry(shape, wcs=None, bbpix=None, tshape=None, dtype=None, comm=None, bbox=None):
	"""Construct a dmap geometry."""
	return DGeometry(shape=shape, wcs=wcs, bbpix=bbpix, tshape=tshape, dtype=dtype, comm=comm, bbox=bbox)

class Dmap(object):
	"""Dmap - distributed enmap. After construction, its relevant members
	are:
		.tiles: list of tile data owned by this task. Each tile is an enmap.
		.work2tile(): sums contribution from all workspaces into tiles
		.tile2work(): projects from tiles to workspaces
		.geometry: geometry and layout of distributed map"""
	def __init__(self, geometry, tiles=None, copy=True):
		try:
			self.geometry = geometry.geometry
			self.tiles = [t.copy() for t in geometry.tiles]
		except AttributeError:
			self.geometry = geometry
			if tiles is not None:
				if copy: tiles = copy=deepcopy(tiles)
				self.tiles = tiles
			else:
				self.tiles = [enmap.empty(ts,tw,dtype=geometry.dtype) for ts,tw in geometry.loc_geometry]
	def work2tile(self, work):
		"""Project from local workspaces into the distributed tiles. Multiple workspaces
		may overlap with a single tile. The contribution from each workspace is summed."""
		self.geometry.work_bufinfo.data2data(work.maps, self.geometry.tile_bufinfo, self.tiles, self.comm)
	def tile2work(self, work=None):
		"""Project from tiles into the local workspaces."""
		if work is None: work = self.geometry.build_work()
		self.geometry.tile_bufinfo.data2data(self.tiles, self.geometry.work_bufinfo, work.maps, self.comm)
		return work
	def copy(self):
		return Dmap(self)
	# Forward some properties to geometry, for convenience
	@property
	def comm(self): return self.geometry.comm
	@property
	def shape(self): return self.geometry.shape
	@property
	def size(self): return self.geometry.size
	@property
	def wcs(self): return self.geometry.wcs
	@property
	def dtype(self): return self.geometry.dtype
	@property
	def ntile(self): return self.geometry.ntile
	@property
	def pre(self): return self.geometry.pre
	@property
	def loc_inds(self): return self.geometry.loc_inds
	@property
	def loc_pos(self): return self.geometry.loc_pos
	@property
	def nloc(self): return self.geometry.nloc
	@property
	def ndim(self): return len(self.shape)
	@property
	def npix(self): return np.product(self.shape[-2:], dtype=int)
	def astype(self, dtype, copy=True):
		"""The copy argument is ignored. It's there for compatibility with
		normal enmaps."""
		if dtype == self.dtype: return self
		else:
			res = Dmap(self.geometry.astype(dtype))
			for st,rt in zip(self.tiles, res.tiles):
				rt[:] = st[:].astype(dtype)
			return res
	def fill(self, val):
		for t in self.tiles: t[:] = val
	def __getitem__(self, sel):
		# Split sel into normal and wcs parts. We only handle non-pixel slices
		sel1, sel2 = utils.split_slice(sel, [self.ndim-2,2])
		if len(sel2) > 0:
			raise NotImplementedError("Pixel slicing of dmaps not implemented")
		geometry = self.geometry[sel1]
		tiles = [tile[sel1] for tile in self.tiles]
		return Dmap(geometry, tiles, copy=False)
	def __setitem__(self, sel, val):
		# Split sel into normal and wcs parts. We only handle non-pixel slices
		sel1, sel2 = utils.split_slice(sel, [self.ndim-2,2])
		if len(sel2) > 0:
			raise NotImplementedError("Pixel slicing of dmaps not implemented")
		try:
			for tile, vtile in zip(self.tiles, val.tiles): tile[sel] = vtile
		except AttributeError:
			for tile in self.tiles: tile[sel] = val
	def __len__(self): return self.shape[0]
	def __repr__(self):
		return "Dmap(%s, rank %d/%d, tiles %d/%d %s)" % (str(self.geometry), self.geometry.comm.rank,
			self.geometry.comm.size, self.geometry.nloc, self.geometry.ntile, str(self.geometry.loc_inds))
	# Math operations
	def _domath(self, other, op, inplace=False):
		res = self if inplace else self.copy()
		try:
			for i, (rtile, otile) in enumerate(zip(res.tiles, other.tiles)):
				res.tiles[i] = op(rtile, otile)
		except AttributeError:
			for i, rtile in enumerate(res.tiles):
				res.tiles[i] = op(rtile, other)
		return res
	def fill_diagonal(self, value):
		"""Fills the diagonal of the non-pixel part of the map with the specified value.
		All pre-dimensions must have the same shape."""
		ndim = len(self.geometry.pre)
		n    = self.geometry.pre[0]
		for tile in self.tiles:
			for i in range(n):
				tile[tuple([i]*ndim)] = value
	def fillbad(self, value=0, inplace=False):
		if not inplace: res = self.copy()
		else: res = self
		for i, tile in enumerate(res.tiles):
			tile[~np.isfinite(tile)] = value
		return res
	def write(self, name, ext="fits", merged=True):
		write_map(name, self, ext=ext, merged=merged)

# Add some math operators
def makefun(key, inplace):
	op = getattr(operator, key)
	def fun(a,b): return a._domath(b, op, inplace=inplace)
	return fun

for opname in ["__add__", "__and__", "__div__", "__eq__", "__floordiv__",
		"__ge__", "__gt__", "__le__", "__lshift__", "__lt__", "__mod__", "__mul__",
		"__ne__", "__or__", "__pow__", "__rshift__", "__sub__", "__truediv__",
		"__xor__"]:
	setattr(Dmap, opname, makefun(opname, False))
for opname in ["__iadd__", "__iand__", "__idiv__", "__ifloordiv__",
		"__ilshift__", "__imod__", "__imul__", "__ior__", "__ipow__",
		"__irshift__", "__isub__", "__itruediv__", "__ixor__"]:
	setattr(Dmap, opname, makefun(opname, True))

def sum(dmap, axis=None):
	"""Sum a dmap along the specified axis, or the flattened version if axis is None.
	The result is a Dmap if the pixel axes are not involved in the sum."""
	# Full sum
	if axis is None:
		return dmap.geometry.comm.allreduce(np.sum([np.sum(t) for t in dmap.tiles]))
	# Non-pixel sums
	if axis < 0: axis = dmap.ndim+axis
	if axis < dmap.ndim-2:
		pre = dmap.pre[:axis]+dmap.pre[axis+1:]
		res = zeros(dmap.geometry.aspre(pre))
		for itile, otile in zip(dmap.tiles, res.tiles):
			otile[:] = np.sum(itile, axis)
		return res
	# Pixel sums: Sum each tile along the specified direction. Then sum tiles
	# that are on the same row/column. Then stack along the remaining row/column
	res = np.zeros(dmap.shape[:axis]+dmap.shape[axis+1:],dmap.dtype)
	paxis = axis-(dmap.ndim-2)
	for tile, ind in zip(dmap.tiles, dmap.loc_inds):
		pbox = dmap.geometry.tile_boxes[ind]
		res[...,pbox[0,1-paxis]:pbox[1,1-paxis]] += np.sum(tile, axis)
	return utils.allreduce(res, dmap.comm)

def broadcast_into(dmap, val, axis=None):
	"""Transpose of a sum a dmap along the specified axis, or the flattened version if axis is None."""
	# Full sum
	if axis is None: dmap[:] = val
	else:
		# Non-pixel sums
		if axis < 0: axis = dmap.ndim+axis
		if axis < dmap.ndim-2:
			for itile, otile in zip(dmap.tiles, val.tiles):
				itile[:] = np.asarray(otile)[(slice(None),)*axis+(None,)+(slice(None),)*(dmap.ndim-axis-1)]
		else:
			# Pixel sums: Sum each tile along the specified direction. Then sum tiles
			# that are on the same row/column. Then stack along the remaining row/column
			res = np.zeros(dmap.shape[:axis]+dmap.shape[axis+1:],dmap.dtype)
			paxis = axis-(dmap.ndim-2)
			for tile, ind in zip(dmap.tiles, dmap.loc_inds):
				pbox = dmap.geometry.tile_boxes[ind]
				if paxis == 0:
					tile[:] = res[Ellipsis,None,pbox[0,1]:pbox[1,1]]
				else:
					tile[:] = res[Ellipsis,pbox[0,0]:pbox[1,0],None]

class DGeometry(object):
	"""DGeometry represents the shape and layout of a DMap."""
	def __init__(self, shape, wcs=None, bbpix=None, tshape=None, dtype=None, comm=None, bbox=None, pre=None):
		"""DGeometry(shape, wcs, bbpix, tshape) or
		DGeometry(dgeometry)

		Construct a DMap geometry. When constructed explicitly from shape, wcs, etc.,
		it is a relatively expensive operation, as MPI communication is necessary.
		Constructing one based on an existing DGeometry is fast.

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
		try:
			# We are pretty lightweight, so copy everything properly. This is
			# less error prone than having changes in one object suddenly affect
			# another. comm must *not* be deepcopied, as that breaks it.
			for key in ["shape","tshape","comm","dtype"]:
				setattr(self, key, getattr(shape, key))
			for key in ["wcs","tile_pos","tile_boxes",
					"tile_geometry","work_geometry",
					"tile_ownership","tile_glob2loc","tile_loc2glob",
					"tile_bufinfo", "work_bufinfo" ]:
				setattr(self, key, copy.deepcopy(getattr(shape, key)))
		except AttributeError:
			# 1. Set up basic properties
			assert shape is not None
			if wcs is None: _, wcs = enmap.geometry(pos=np.array([[-1,-1],[1,1]])*5*np.pi/180, shape=shape[-2:])
			if comm is None: comm = mpi.COMM_WORLD
			if tshape is None: tshape = (720,720)
			if dtype is None: dtype = np.float64
			if bbpix is None:
				if bbox is None: bbpix = [[0,0],shape[-2:]]
				else: bbpix = box2pix(shape, wcs, bbox)
			nphi  = int(np.round(np.abs(360/wcs.wcs.cdelt[0])))
			# Reorder from/to if necessary, and expand to [:,2,2]
			bbpix = sanitize_pixbox(bbpix, shape)
			dtype = np.dtype(dtype)
			self.shape,  self.wcs,   self.bbpix = tuple(shape),  wcs.deepcopy(), np.array(bbpix)
			self.tshape, self.dtype, self.comm  = tuple(tshape), dtype, comm

			# 2. Set up workspace descriptions
			work_geometry = [enmap.slice_geometry(shape, wcs, (slice(b[0,0],b[1,0]),slice(b[0,1],b[1,1])), nowrap=True) for b in bbpix]
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
			tile_indices = np.array([np.arange(ntile)/bshape[1],np.arange(ntile)%bshape[1]]).T
			# 4. Define tile ownership.
			# a) For each task compute the overlap of each tile with its workspaces, and
			#    concatenate across tasks to form a [nworktot,ntile] array.
			wslices = select_nonempty( # slices into work
					utils.allgatherv(utils.box_slice(bbpix, tbox),comm, axis=0), # normal
					utils.allgatherv(utils.box_slice(bbpix+[0,nphi], tbox),comm, axis=0)) # wrapped
			tslices = select_nonempty( # slices into tiles
					utils.allgatherv(utils.box_slice(tbox, bbpix),comm, axis=1), # normal
					utils.allgatherv(utils.box_slice(tbox, bbpix+[0,nphi]),comm, axis=1)) # wrapped
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
			tile_geometry = [enmap.slice_geometry(shape, wcs, (slice(tb[0,0],tb[1,0]),slice(tb[0,1],tb[1,1]))) for tb in tbox]
			# 6. Define mapping between work<->wbuf and tiles<->tbuf
			wbufinfo  = np.zeros([2,comm.size],dtype=int)
			tbufinfo  = np.zeros([2,comm.size],dtype=int)
			winfo, tinfo = [], []
			woff, toff = 0, 0
			prelen = np.product(shape[:-2], dtype=int)
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
			# TODO: store tbox? loc_geometry vs. tile_geometry, etc.
			# 7. Store
			# [ntile,2]: position of each (global) tile in grid
			self.tile_pos   = tile_indices
			# [ntile,2,2]: pixel box for each (global) tile
			self.tile_boxes = tbox
			# [ntile,(shape,wcs)]: geometry of each (global) tile
			self.tile_geometry= tile_geometry
			# [nwork,(shape,wcs)]: geometry of each local workspace
			self.work_geometry= work_geometry
			# [ntile]: rank of owner of each tile
			self.tile_ownership = town
			# [ntile]: local index of each (global) tile
			self.tile_glob2loc  = tlmap
			# [nloc]:  global index of each local tile
			self.tile_loc2glob  = tgmap[comm.rank]
			# Communication buffers
			self.tile_bufinfo = Bufmap(tinfo, tbufinfo, toff)
			self.work_bufinfo = Bufmap(winfo, wbufinfo, woff)
	@property
	def pre(self): return self.shape[:-2]
	@pre.setter
	def pre(self, val):
		"""Change the shape of the non-pixel dimensions. Both longer and shorter vals are supported."""
		try: val = tuple(val)
		except TypeError: val = (val,)
		oldlen = np.product(self.pre, dtype=int)
		self.shape = tuple(val)+self.shape[-2:]
		newlen = np.product(self.pre, dtype=int)
		# These are affected by non-pixel slicing:
		# shape, tile_geometry, work_geometry, tile_bufinfo, work_bufinfo
		# Bufinfos change due to the different amount of data involved
		self.tile_geometry = [(self.pre+ts[-2:],tw) for ts,tw in self.tile_geometry]
		self.work_geometry = [(self.pre+ws[-2:],ww) for ws,ww in self.work_geometry]
		self.tile_bufinfo  = self.tile_bufinfo.slice_helper(newlen, oldlen)
		self.work_bufinfo  = self.work_bufinfo.slice_helper(newlen, oldlen)
	@property
	def size(self): return np.product(self.shape, dtype=int)
	@property
	def npix(self): return np.product(self.shape[-2:], dtype=int)
	@property
	def ndim(self): return len(self.shape)
	@property
	def ntile(self): return len(self.tile_ownership)
	@property
	def loc_inds(self): return self.tile_loc2glob
	@property
	def loc_pos(self): return self.tile_pos[self.loc_inds]
	@property
	def loc_geometry(self): return [self.tile_geometry[i] for i in self.tile_loc2glob]
	@property
	def nloc(self): return len(self.loc_inds)
	def __repr__(self):
		return "DGeometry(shape=%s,wcs=%s,bbpix=%s,tshape=%s,dtype=%s,comm=%s)" % (self.shape, self.wcs, str(self.bbpix).replace("\n",","), self.tshape, np.dtype(self.dtype).name, self.comm.name)
	def copy(self): return DGeometry(self)
	def astype(self, dtype):
		if dtype == self.dtype: return self
		res = self.copy()
		res.dtype = dtype
		return res
	def aspre(self, pre):
		if pre == self.pre: return self
		res = self.copy()
		res.pre = pre
		return res
	def build_work(self):
		return Workspace([enmap.zeros(ws,ww,dtype=self.dtype) for ws,ww in self.work_geometry])
	def __getitem__(self, sel):
		# Split sel into normal and wcs parts.
		sel1, sel2 = utils.split_slice(sel, [self.ndim-2,2])
		if len(sel2) > 0: raise NotImplementedError("Pixel slicing of dmap geometries not implemented")
		res = self.copy()
		res.pre= np.zeros(self.pre)[sel].shape
		return res

class Workspace:
	def __init__(self, maps):
		self.maps = maps
	def copy(self): return Workspace([m.copy() for m in self.maps])
	def __getitem__(self, i): return Workspace([m[i] for m in self.maps])
	def __setitem__(self, i, maps):
		if isinstance(maps, Workspace):
			for mi, mo in zip(self.maps, maps):
				mi[i] = mo
		else:
			for mi in self.maps:
				mi[i] = maps
	def _domath(self, other, op, inplace=False):
		res = self if inplace else self.copy()
		if isinstance(other, Workspace):
			for mi, mo in zip(res.maps, other.maps):
				mi[:] = op(mi, mo)
		else:
			for mi in res.maps:
				mi[:] = op(mi, other)
		return res
	
for opname in ["__add__", "__and__", "__div__", "__eq__", "__floordiv__",
		"__ge__", "__gt__", "__le__", "__lshift__", "__lt__", "__mod__", "__mul__",
		"__ne__", "__or__", "__pow__", "__rshift__", "__sub__", "__truediv__",
		"__xor__"]:
	setattr(Workspace, opname, makefun(opname, False))
for opname in ["__iadd__", "__iand__", "__idiv__", "__ifloordiv__",
		"__ilshift__", "__imod__", "__imul__", "__ior__", "__ipow__",
		"__irshift__", "__isub__", "__itruediv__", "__ixor__"]:
	setattr(Workspace, opname, makefun(opname, True))

class Bufmap:
	"""This class encapsulates the information needed to transer data
	between a list of numpy arrays and an mpi alltoallv buffer, as well
	as the parameters for alltoallv. It is a helper class for DGeometry
	and DMap. It does not handle the scomplicated construction of
	the info objects itself."""
	def __init__(self, data_info, buf_info=None, buf_shape=None):
		try:
			for key in ["data_info","buf_info","buf_shape"]:
					setattr(self, key, copy(getattr(data_info, key)))
		except AttributeError:
			assert data_info is not None
			assert buf_info  is not None
			assert buf_shape is not None
			self.data_info, self.buf_info, self.buf_shape = copy.deepcopy(data_info), copy.deepcopy(buf_info), copy.deepcopy(buf_shape)
	def data2buf(self, data, buf):
		"""Transfer data to a buffer compatible with this bufmap."""
		for ind, dslice, bslice in self.data_info:
			buf[bslice] = data[ind][dslice].reshape(-1)
	def buf2data(self, buf, data):
		"""Transfer data from a buffer compatible with this bufmap.
		Parts of data that are hit multiple times are accumulated."""
		for d in data: d[:] = 0
		for ind, dslice, bslice in self.data_info:
			data[ind][dslice] += buf[bslice].reshape(data[ind][dslice].shape)
	def buf2buf(self, source_buf, target_bufmap, target_buf, comm):
		"""Transfer data from one buffer to another using MPI."""
		comm.Alltoallv((source_buf, self.buf_info),(target_buf,target_bufmap.buf_info))
	def data2data(self, source_data, target_bufmap, target_data, comm):
		"""Transfer data from one configuration (as described by this
		bufmap) to another (as described by target_bufmap), allocating
		buffers internally as needed."""
		# Use dtype.name here to work around mpi4py's inability to handle
		# numpy's several equivalent descriptions of the same dtype. This
		# prevents errors like "KeyError '<f'"
		source_buffer = np.zeros(self.buf_shape, np.dtype(source_data[0].dtype).name)
		target_buffer = np.zeros(target_bufmap.buf_shape, np.dtype(target_data[0].dtype).name)
		self.data2buf(source_data, source_buffer)
		self.buf2buf(source_buffer, target_bufmap, target_buffer, comm)
		target_bufmap.buf2data(target_buffer, target_data)
	def slice_helper(self, newlen, oldlen):
		"""Returns a new Bufmap with buffer slices scaled by newlen/oldlen.
		This is used for defining slice operations."""
		buf_info  = tuple(np.array(self.buf_info)*newlen/oldlen)
		buf_shape = self.buf_shape*newlen/oldlen
		data_info = [(ind,dslice,slice(bslice.start*newlen/oldlen,bslice.stop*newlen/oldlen)) for ind,dslice,bslice in self.data_info]
		return Bufmap(data_info, buf_info, buf_shape)

def write_map(name, map, ext="fits", merged=True):
	if not merged:
		# Write as individual tiles in directory of the specified name
		utils.mkdir(name)
		for pos, tile in zip(map.loc_pos,map.tiles):
			enmap.write_map(name + "/tile%03d_%03d.%s" % (tuple(pos)+(ext,)), tile)
	else:
		# Write to a single file. This currently creates the full map
		# in memory while writing. It is unclear how to avoid this
		# without bypassing pyfits or becoming super-slow.
		if map.comm.rank == 0:
			canvas = enmap.zeros(map.shape, map.wcs, np.dtype(map.dtype).name)
		else:
			canvas = None
		dmap2enmap(map, canvas)
		if map.comm.rank == 0:
			enmap.write_map(name, canvas)

def read_map(name, bbpix=None, bbox=None, tshape=None, comm=None, pixbox=None):
	if comm is None: comm = mpi.COMM_WORLD
	if os.path.isdir(name):
		if pixbox is not None: raise NotImplementedError("dmap.read_map with a pixbox is not implemented for dmaps that are stored as tiles on disk")
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
		map = Dmap(geometry(shape, wcs, bbpix=bbpix, bbox=bbox, tshape=tshape, dtype=dtype, comm=comm))
		for pos, tile in zip(map.loc_pos,map.tiles):
			tile[:] = enmap.read_map(name+"/"+tfiles[pos[0]][pos[1]])
	else:
		# Map is in a single file. This is very memory-wasteful - 30 GB for an advact map
		# Get map info
		if comm.rank == 0:
			canvas = enmap.read_map(name, pixbox=pixbox)
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
		map = Dmap(geometry(shape, wcs, bbpix=bbpix, bbox=bbox, tshape=tshape, dtype=dtype, comm=comm))
		# And send data to the tiles
		enmap2dmap(canvas, map)
	return map

def enmap2dmap(emap, dmap, root=0):
	"""Import data from an enmap into a dmap."""
	for ti in range(dmap.ntile):
		id  = dmap.geometry.tile_ownership[ti]
		loc = dmap.geometry.tile_glob2loc[ti]
		box = dmap.geometry.tile_boxes[ti]
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
		id  = dmap.geometry.tile_ownership[ti]
		loc = dmap.geometry.tile_glob2loc[ti]
		box = dmap.geometry.tile_boxes[ti]
		if dmap.comm.rank == root and id == root:
			data = dmap.tiles[loc]
		elif dmap.comm.rank == root:
			data = np.zeros(dmap.pre+tuple(box[1]-box[0]), dtype=np.dtype(dmap.dtype).name)
			dmap.comm.Recv(data, source=id, tag=loc)
		elif dmap.comm.rank == id:
			dmap.comm.Send(dmap.tiles[loc], dest=root, tag=loc)
		if dmap.comm.rank == root:
			emap[...,box[0,0]:box[1,0],box[0,1]:box[1,1]] = data

def box2pix(shape, wcs, box):
	"""Convert one or several bounding boxes of shape [2,2] or [n,2,2]
	into pixel counding boxes in standard python half-open format.
	The box must be [{from,to},...].
	"""
	nphi = int(np.round(np.abs(360./wcs.wcs.cdelt[0])))
	box  = np.asarray(box)
	fbox = box.reshape(-1,2,2)
	if wcs.wcs.cdelt[0] < 0: fbox[...,1] = fbox[...,::-1,1]
	# Must rollaxis because sky2pix expects [{dec,ra},...]
	ibox = enmap.sky2pix(shape, wcs, utils.moveaxis(fbox,2,0), corner=True)
	# FIXME: This is one of many places in the code that will break if a bounding
	# box goes more than half the way around the sky. Properly fixing
	# this will require a big overhaul, especially if we want to handle
	# bounding boxes that are bigger than the full sky.
	ibox[1] = utils.unwrap_range(ibox[1].T, nphi).T
	#ibox[1] = utils.unwind(ibox[1], nphi)
	# We now have [{y,x},:,{from,to}
	# Truncate to integer, and add one to endpoint to make halfopen interval
	ibox = np.floor(np.sort(ibox, 2)).astype(int)
	# Add 1 to endpoint to make halfopen interval
	ibox[:,:,1] += 1
	#ibox = np.array([np.floor(ibox[0]),np.ceil(ibox[1])]).astype(int)
	# Shuffle to [:,{from,to},{y,x}]
	ibox = utils.moveaxis(ibox, 0, 2)
	return ibox.reshape(box.shape)

def sanitize_pixbox(bbpix, shape):
	bbpix = np.array(bbpix)
	# For slicing, we need the first bound to be lower than the second bound.
	# It might not be, as Ra and x have opposite ordering. So sort along the
	# from-to axis. bbpix has 3 dims [scan,fromto,radec]
	bbpix = bbpix.reshape((-1,)+bbpix.shape[-2:])
	bbpix = np.sort(bbpix, 1)
	# Some scans may extend partially beyond the end of our map. We must therefore
	# truncate the bounding box.
	#bbpix[:,0,:] = np.maximum(bbpix[:,0,:],0)
	#bbpix[:,1,:] = np.minimum(bbpix[:,1,:],shape[-2:])
	return bbpix

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
	if len(boxes) < nsplit:
		# If we have fewer tods than processes, just assign one to each, and give empty
		# ones to the remainder
		return [[[i]] for i in range(len(boxes))] + [[[]] for i in range(len(boxes),nsplit)]
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

def select_nonempty(a, b):
	asize = np.product(a[...,1,:]-a[...,0,:],-1,dtype=int)
	return np.where(asize[...,None,None] > 0, a, b)
