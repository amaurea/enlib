import numpy as np, glob, re, sys, os
from enlib import utils, enmap, bunch

def leaftile(idir, odir, tsize=675, comm=None, verbose=False, lrange=[0,-6],
		monolithic=False):
	"""Given a input directory containing a tiled dmap in standard
	ordering, outputs a leaflet-compatible hierarchy of tiles in
	odir with tile size tsize."""
	# First create our base tiles. These have opposite y ordering than
	# dmap tiles, and may be different-sized.
	otilename = "tile_%(y)d_%(x)d.fits"
	if not monolithic:
		itilename = idir + "/tile%(y)03d_%(x)03d.fits"
		itile1, itile2 = None, None
	else:
		itilename = idir
		itile1, itile2 = (0,0), (1,1)
	retile(itilename,
			"%s/%d/%s" % (odir,lrange[0],otilename),
			ocorner=(np.pi/2,-np.pi), otilesize=(-tsize,tsize),
			comm=comm, verbose=verbose, itile1=itile1, itile2=itile2)
	# Then loop over the smaller levels
	for level in range(lrange[0]-1,lrange[1],-1):
		if comm: comm.barrier()
		combine_tiles("%s/%d/%s" % (odir, level+1, otilename),
				"%s/%d/%s" % (odir, level, otilename), tyflip=True,
				pad_to=tsize, comm=comm, verbose=verbose)

def combine_tiles(ipathfmt, opathfmt, combine=2, downsample=2,
		itile1=(None,None), itile2=(None,None), tyflip=False, txflip=False,
		pad_to=None, comm=None, verbose=False):
	"""Given a set of tiles on disk at locaiton ipathfmt % {"y":...,"x"...},
	combine them into larger tiles, downsample and write the result to
	opathfmt % {"y":...,"x":...}. x and y must be contiguous and start at 0.
	
	reftile[2] indicates the tile coordinates of the first valid input tile.
	This needs to be specified if not all tiles of the logical tiling are
	physically present.

	tyflip and txflip indicate if the tiles coordinate system is reversed
	relative to the pixel coordinates or not."
	"""
	# Expand combine and downsample to 2d
	combine    = np.zeros(2,int)+combine
	downsample = np.zeros(2,int)+downsample
	if pad_to is not None:
		pad_to = np.zeros(2,int)+pad_to
	# Handle optional mpi
	rank, size = (comm.rank, comm.size) if comm is not None else (0, 1)
	# Find the range of input tiles
	itile1, itile2 = find_tile_range(ipathfmt, itile1, itile2)
	# Read the first tile to get its size information
	ibase = enmap.read_map(ipathfmt % {"y":itile1[0],"x":itile1[1]})*0
	# Find the set of output tiles we need to consider
	otile1 = itile1/combine
	otile2 = (itile2-1)/combine+1
	# And loop over them
	oyx = [(oy,ox) for oy in range(otile1[0],otile2[0]) for ox in range(otile1[1],otile2[1])]
	for i in range(rank, len(oyx), size):
		oy, ox = oyx[i]
		# Read in all associated tiles into a list of lists
		rows = []
		for dy in range(combine[0]):
			iy = oy*combine[0] + dy
			if iy >= itile2[0]: continue
			cols = []
			for dx in range(combine[1]):
				ix = ox*combine[1] + dx
				if ix >= itile2[1]: continue
				if iy < itile1[0] or ix < itile1[1]:
					# The first tiles are missing on disk, but are
					# logically a part of the tiling. Use ibase,
					# which has been zeroed out.
					cols.append(ibase)
				else:
					itname = ipathfmt % {"y": iy, "x": ix}
					cols.append(enmap.read_map(itname))
			if txflip: cols = cols[::-1]
			rows.append(cols)
		# Stack them next to each other into a big tile
		if tyflip: rows = rows[::-1]
		omap = enmap.tile_maps(rows)
		# Downgrade if necessary
		if np.any(downsample>1):
			omap = enmap.downgrade(omap, downsample)
		if pad_to is not None:
			# Padding happens towards the end of the tiling,
			# which depends on the flip status
			padding = np.array([[0,0],[pad_to[0]-omap.shape[-2],pad_to[1]-omap.shape[-1]]])
			if tyflip: padding[:,0] = padding[::-1,0]
			if txflip: padding[:,1] = padding[::-1,1]
			omap = enmap.pad(omap, padding)
		# And output
		otname = opathfmt % {"y": oy, "x": ox}
		utils.mkdir(os.path.dirname(otname))
		enmap.write_map(otname, omap)
		if verbose: print otname

def retile(ipathfmt, opathfmt, itile1=(None,None), itile2=(None,None),
		otileoff=(0,0), otilenum=(None,None), ocorner=(-np.pi/2,-np.pi),
		otilesize=(675,675), comm=None, verbose=False):
	"""Given a set of tiles on disk with locations ipathfmt % {"y":...,"x":...},
	retile them into a new tiling and write the result to opathfmt % {"y":...,"x":...}.
	The new tiling will have tile size given by otilesize[2]. Negative size means the
	tiling will to down/left instead of up/right. The corner of the tiling will
	be at sky coordinates ocorner[2] in radians. The new tiling will be pixel-
	compatible with the input tiling - w.g. the wcs will only differ by crpix.

	The output tiling will logically cover the whole sky, but only output tiles
	that overlap with input tiles will actually be written. This can be modified
	by using otileoff[2] and otilenum[2]. otileoff gives the tile indices of the
	corner tile, while otilenum indicates the number of tiles to write."""
	# Set up mpi
	rank, size = (comm.rank, comm.size) if comm is not None else (0, 1)
	# Expand any scalars
	otilesize = np.zeros(2,int)+otilesize
	otileoff  = np.zeros(2,int)+otileoff
	# Find the range of input tiles
	itile1, itile2 = find_tile_range(ipathfmt, itile1, itile2)
	# To fill in the rest of the information we need to know more
	# about the input tiling, so read the first tile
	ibase = enmap.read_map(ipathfmt % {"y":itile1[0],"x":itile1[1]})
	itilesize = ibase.shape[-2:]
	# Find the pixel position of our output corners according to the wcs.
	# This is the last place we need to do a coordinate transformation.
	# All the rest can be done in pure pixel logic.
	pixoff = np.round(ibase.sky2pix(ocorner)).astype(int)
	# Find the range of output tiles
	def pix2otile(pix, ioff, osize): return (pix-ioff)/osize
	otile1 = pix2otile(itile1*itilesize,   pixoff, otilesize)
	otile2 = pix2otile(itile2*itilesize-1, pixoff, otilesize)
	otile1, otile2 = np.minimum(otile1,otile2), np.maximum(otile1,otile2)
	otile2 += 1
	# We can now loop over output tiles
	cache = [None,None,None]
	oyx = [(oy,ox) for oy in range(otile1[0],otile2[0]) for ox in range(otile1[1],otile2[1])]
	for i in range(rank, len(oyx), size):
		otile = np.array(oyx[i])
		# Find out which input tiles overlap with this output tile.
		# Our tile stretches from opix1:opix2 relative to the global input pixels
		opix1 = otile*otilesize + pixoff
		opix2 = (otile+1)*otilesize + pixoff
		# output tiles and input tiles may increase in opposite directions
		opix1, opix2 = np.minimum(opix1,opix2), np.maximum(opix1,opix2)
		try: omap = read_area(ipathfmt, [opix1,opix2],itile1=itile1, itile2=itile2,cache=cache)
		except IOError: continue
		oname = opathfmt % {"y":otile[0]+otileoff[0],"x":otile[1]+otileoff[1]}
		utils.mkdir(os.path.dirname(oname))
		enmap.write_map(oname, omap)
		if verbose: print oname

def monolithic(idir, ofile, verbose=True):
	# Find the range of input tiles
	ipathfmt = idir + "/tile%(y)03d_%(x)03d.fits"
	itile1, itile2 = find_tile_range(ipathfmt)
	# Read the first and last tile to get the total dimensions
	m1 = enmap.read_map(ipathfmt % {"y":itile1[0],"x":itile1[1]})
	m2 = enmap.read_map(ipathfmt % {"y":itile2[0]-1,"x":itile2[1]-1})
	wy,wx  = m1.shape[-2:]
	oshape = tuple(np.array(m1.shape[-2:])*(itile2-itile1-1) + np.array(m2.shape[-2:]))
	omap  = enmap.zeros(m1.shape[:-2] + oshape, m1.wcs, m1.dtype)
	del m1, m2
	# Now loop through all tiles and copy them in to the correct position
	for ty in range(itile1[0],itile2[0]):
		for tx in range(itile1[1],itile2[1]):
			m = enmap.read_map(ipathfmt % {"y":ty,"x":tx})
			omap[...,ty*wy:(ty+1)*wy,tx*wx:(tx+1)*wx] = m
			if verbose: print ipathfmt % {"y":ty,"x":tx}
	enmap.write_map(ofile, omap)

def range_overlap(a,b):
	return np.array([np.maximum(a[0],b[0]),np.minimum(a[1],b[1])])

def find_tile_range(pathfmt, tile1=(None,None), tile2=(None,None)):
	"""Given a path format with with labeled formats including y
	and x (like %(y)d), Return the range of tiles available, in
	the form of [{start,end},{y,x}] tile indices.
	If tile1[2] is specified, then this will override the start
	of the result. If tile2[2] is specified, then it will override
	the end of the result."""
	if tile1 is None: tile1 = (None,None)
	if tile2 is None: tile2 = (None,None)
	# If both offset and ntile are specified, we don't need any
	# complicated disk search.
	if tile1[0] is not None and tile1[1] is not None and tile2[0] is not None and tile2[1] is not None: return np.asarray(tile1), np.asarray(tile2)
	# Find the min/max on disk. We do that by constructing a glob to
	# roughly match them, and them filtering them with a regex.
	ranges = [None,None]
	gstr  = utils.format_to_glob(pathfmt)
	regex = utils.format_to_regex(pathfmt)
	files = glob.glob(gstr)
	if len(files) == 0:
		raise ValueError("Found no files matching path format!")
	for file in files:
		m = re.match(regex, file)
		if not m: continue
		yx = [int(m.group(name)) for name in ["y","x"]]
		for i in range(2):
			if ranges[i] is None: ranges[i] = [yx[i],yx[i]+1]
			ranges[i] = [min(ranges[i][0],yx[i]),max(ranges[i][1],yx[i]+1)]
	# Override if needed
	for i in range(2):
		if tile1[i] is not None:
			ranges[i][0] = tile1[i]
		if tile2[i]  is not None:
			ranges[i][1] = tile2[i]
	ranges = np.array(ranges)
	return ranges[:,0], ranges[:,1]

def read_tileset_geometry(ipathfmt, itile1=(None,None), itile2=(None,None)):
	itile1, itile2 = find_tile_range(ipathfmt, itile1, itile2)
	m1 = enmap.read_map(ipathfmt % {"y":itile1[0],"x":itile1[1]})
	m2 = enmap.read_map(ipathfmt % {"y":itile2[0]-1,"x":itile2[1]-1})
	wy,wx  = m1.shape[-2:]
	oshape = tuple(np.array(m1.shape[-2:])*(itile2-itile1-1) + np.array(m2.shape[-2:]))
	return bunch.Bunch(shape=m1.shape[:-2]+oshape, wcs=m1.wcs, dtype=m1.dtype,
			tshape=m1.shape[-2:])

def read_area(ipathfmt, opix, itile1=(None,None), itile2=(None,None), verbose=False,
		cache=None):
	"""Given a set of tiles on disk with locations ipathfmt % {"y":...,"x":...},
	read the data corresponding to the pixel range opix[{from,to],{y,x}] in
	the full map."""
	opix = np.asarray(opix)
	# Find the range of input tiles
	itile1, itile2 = find_tile_range(ipathfmt, itile1, itile2)
	# To fill in the rest of the information we need to know more
	# about the input tiling, so read the first tile
	if cache is None or cache[2] is None:
		geo = read_tileset_geometry(ipathfmt, itile1=itile1, itile2=itile2)
	else: geo = cache[2]
	if cache is not None: cache[2] = geo
	isize = geo.tshape
	osize = opix[1]-opix[0]
	omap  = enmap.zeros(geo.shape[:-2]+tuple(osize), geo.wcs, geo.dtype)
	# Find out which input tiles overlap with this output tile.
	# Our tile stretches from opix1:opix2 relative to the global input pixels
	it1 = opix[0]/isize
	it2 = (opix[1]-1)/isize+1
	noverlap = 0
	for ity in range(it1[0],it2[0]):
		if ity < itile1[0] or ity >= itile2[0]: continue
		# Start/end of this tile in global input pixels
		ipy1, ipy2 = ity*isize[0], (ity+1)*isize[0]
		overlap = range_overlap(opix[:,0],[ipy1,ipy2])
		oy1,oy2 = overlap-opix[0,0]
		iy1,iy2 = overlap-ipy1
		for itx in range(it1[1],it2[1]):
			if itx < itile1[1] or itx >= itile2[1]: continue
			ipx1, ipx2 = itx*isize[1], (itx+1)*isize[1]
			overlap = range_overlap(opix[:,1],[ipx1,ipx2])
			ox1,ox2 = overlap-opix[0,1]
			ix1,ix2 = overlap-ipx1
			# Read the input tile and copy over
			iname = ipathfmt % {"y":ity,"x":itx}
			if cache is None or cache[0] != iname:
				imap  = enmap.read_map(iname)
			else: imap = cache[1]
			if cache is not None:
				cache[0], cache[1] = iname, imap
			if verbose: print iname
			# Edge input tiles may be smaller than the standard
			# size.
			ysub = isize[0]-imap.shape[-2]
			xsub = isize[1]-imap.shape[-1]
			# If the input map is too small, there may actually be
			# zero overlap.
			if oy2-ysub <= oy1 or ox2-xsub <= ox1: continue
			omap[...,oy1:oy2-ysub,ox1:ox2-xsub] = imap[...,iy1:iy2-ysub,ix1:ix2-xsub]
			noverlap += 1
	if noverlap == 0:
		raise IOError("No tiles for tiling %s in range %s" % (ipathfmt, ",".join([":".join([str(p) for p in r]) for r in opix.T])))
	# Set up the wcs for the output tile
	omap.wcs.wcs.crpix -= opix[0,::-1]
	return omap

def read_retile(ipathfmt, tpos, otilesize=None, pixoff=(0,0), margin=0,
		itile1=(None,None), itile2=(None,None), verbose=False):
	"""Read a single tile from the tiling at ipathfmt % {"y":tpos[0],"x":tpos[1]},
	returning it as an enmap. If otilesize or pixoff are specified, then
	the tiling will be from a retiling with that tile size and pixel offset,
	which means that behind the scenes multiple tiles will be read and stitched
	together. If margin[{left,right},{y,x}] is not zero, then it specifies the
	number of pixels to extend the tile by. The tile will be extended with
	data from the tiling, i.e. from the neighboring tiles."""
	if otilesize is None:
		geo = read_tileset_geometry(ipathfmt, itile1, itile2)
		otilesize = geo.tshape
	tpos   = np.zeros(2,int)+tpos
	pixoff = np.zeros(2,int)+pixoff
	margin = np.zeros((2,2),int)+margin
	pixbox = np.array([tpos*otilesize,(tpos+1)*otilesize])+pixoff
	pixbox[0] -= margin[0]
	pixbox[1] += margin[1]
	return read_area(ipathfmt, pixbox, itile1=itile1, itile2=itile2, verbose=verbose)

def retile_iterator(ipathfmt, otilesize=None, pixoff=(0,0), margin=0,
		itile1=(None,None), itile2=(None,None), comm=None, verbose=False):
	"""Iterator that yields a series of tiles from the tileset given by
	ipathfmt. See read_retile for how margin, otilesize and pixoff
	allow you to iterate over a modified tiling. This currently just
	calls read_area repeatedly, so it is not as efficient as it could have
	been."""
	# Handle mpi
	rank, nproc = (0,1) if comm is None else (comm.rank, comm.size)
	# Find the number of tiles to iterate over
	geo = read_tileset_geometry(ipathfmt, itile1, itile2)
	if otilesize is None: otilesize = geo.tshape
	otilesize = np.array(otilesize)
	notile = (np.array(geo.shape[-2:])-pixoff+otilesize-1)/otilesize
	tyx = [(y,x) for y in range(notile[0]) for x in range(notile[1])]
	for i in range(rank, len(tyx), nproc):
		tpos = tyx[i]
		yield tpos, read_retile(ipathfmt, tpos, otilesize=otilesize, pixoff=pixoff,
				margin=margin, itile1=itile1, itile2=itile2)
