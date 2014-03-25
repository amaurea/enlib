import numpy as np, scipy.ndimage, warnings, enlib.wcs, enlib.slice, astropy.io.fits, h5py

# PyFits uses row-major ordering, i.e. C ordering, while the fits file
# itself uses column-major ordering. So an array which is (ncomp,ny,nx)
# will be (nx,ny,ncomp) in the file. This means that the axes in the ndmap
# will be in the opposite order of those in the wcs object.

class ndmap(np.ndarray):
	"""Implements (stacks of) flat, rectangular, 2-dimensional maps as a dense
	numpy array with a fits WCS. The axes have the reverse ordering as in the
	fits file, and hence the WCS object. This class is usually constructed by
	using one of the functions following it, much like numpy arrays. We assume
	that the WCS only has two axes with unit degrees. The ndmap itself uses
	radians for everything."""
	def __new__(cls, arr, wcs):
		"""Wraps a numpy and bounding box into an ndmap."""
		obj = np.asarray(arr).view(cls)
		obj.wcs = wcs.deepcopy()
		return obj
	def __array_finalize__(self, obj):
		if obj is None: return
		self.wcs = getattr(obj, "wcs", None)
	def __repr__(self):
		return "ndmap(%s,%s)" % (np.asarray(self), enlib.wcs.describe(self.wcs))
	def __getitem__(self, sel):
		return np.ndarray.__getitem__(self, sel)
	def __array_wrap__(self, arr, context=None):
		if arr.ndim < 2: return arr
		return ndmap(arr, self.wcs)
	def copy(self):
		return ndmap(np.array(self), self.wcs)
	def sky2pix(self, coords, safe=True): return sky2pix(self.wcs, coords, safe)
	def pix2sky(self, pix,    safe=True): return pix2sky(self.wcs, pix,    safe)
	def box(self): return box(self.shape, self.wcs)
	def posmap(self): return posmap(self.shape, self.wcs)
	def freqmap(self): return freqmap(self.shape, self.wcs)
	def lmap(self): return lmap(self.shape, self.wcs)
	def area(self): return area(self.shape, self.wcs)
	def project(self, shape, wcs, order=3): return project(self, shape, wcs, order)
	def __getitem__(self, sel):
		# Split sel into normal and wcs parts.
		sel1, sel2 = enlib.slice.split_slice(sel, [self.ndim-2,2])
		# No index creation supported in the wcs part
		if any([s is None for s in sel2]):
			raise IndexError("None-indices not supported for the wcs part of an ndmap.")
		if len(sel2) > 2:
			raise IndexError("too many indices")
		# If the wcs slice includes direct indexing, so that wcs
		# axes are lost, then degrade to a normal numpy array,
		# since this class assumes that the two last axes are
		# wcs axes.
		if any([type(s) is int for s in sel2]):
			return np.asarray(self)[sel]
		# Otherwise we will return a full ndmap, including a
		# (possibly) sliced wcs.
		wcs = self.wcs.deepcopy()
		if len(sel2) > 0:
			# The wcs object has the indices in reverse order
			for i,s in enumerate(sel2[::-1]):
				s = enlib.slice.expand_slice(s, self.shape[::-1][-2+i])
				wcs.wcs.crpix[i] -= s.start
				wcs.wcs.crpix[i] /= s.step
				wcs.wcs.cdelt[i] *= s.step
		return ndmap(np.ndarray.__getitem__(self, sel), wcs)
	def __getslice__(self, a, b=None, c=None): return self[slice(a,b,c)]
	def submap(self, box, inclusive=False):
		"""submap(box, inclusive=False)
		
		Extract the part of the map inside the given box.

		Parameters
		----------
		box : array_like
			The [[fromy,fromx],[toy,tox]] bounding box to select.
			The resulting map will have a bounding box as close
			as possible to this, but will differ slightly due to
			the finite pixel size.
		inclusive : boolean
			Whether to include pixels that are only partially
			inside the bounding box. Default: False."""
		ibox = self.subinds(box, inclusive)
		return self[...,ibox[0,0]:ibox[1,0],ibox[0,1]:ibox[1,1]]
	def subinds(self, box, inclusive=False):
		"""Helper function for submap. Translates the bounding
		box provided into a pixel units. Assumes rectangular
		coordinates."""
		box  = np.asarray(box)
		# Translate the box to pixels. The 0.5 moves us from
		# pixel-center coordinates to pixel-edge coordinates.
		bpix = self.wcs.wcs_world2pix(box[:,::-1]*180/np.pi,0)[:,::-1]+0.5
		# If we are inclusive, find a bounding box, otherwise,
		# an internal box
		if inclusive:
			ibox = np.array([np.floor(bpix[0]),np.ceil(bpix[1])],dtype=int)
		else:
			ibox = np.array([np.ceil(bpix[0]),np.floor(bpix[1])],dtype=int)
		return ibox

def box(shape, wcs):
	pix    = np.array([[0,0],shape[-2:]]).T-0.5
	coords = wcs.wcs_pix2world(pix[1],pix[0],0)[::-1]
	return enlib.utils.unwind(np.array(coords).T*np.pi/180)

def enmap(arr, wcs=None, dtype=None, copy=True):
	"""Construct an ndmap from data.

	Parameters
	----------
	arr : array_like
		The data to initialize the map with.
		Must be at least two-dimensional.
	wcs : WCS object
	dtype : data-type, optional
		The data type of the map.
		Default: Same as arr.
	copy : boolean
		If true, arr is copied. Otherwise, a referance is kept."""
	if copy:
		arr = np.array(arr, dtype=dtype)
	if wcs is None:
		if isinstance(arr, ndmap):
			wcs = arr.wcs
		else:
			return arr
	return ndmap(arr, wcs)

def zeros(shape, wcs, dtype=None):
	return enmap(np.zeros(shape), wcs, dtype=dtype)
def empty(shape, wcs, dtype=None):
	return enmap(np.empty(shape), wcs, dtype=dtype)

def posmap(shape, wcs, safe=True):
	"""Return an enmap where each entry is the coordinate of that entry,
	such that posmap(shape,wcs)[{0,1},j,k] is the {y,x}-coordinate of
	pixel (j,k) in the map. Results are returned in radians, and
	if safe is true (default), then sharp coordinate edges will be
	avoided."""
	pix    = np.mgrid[:shape[-2],:shape[-1]]
	return ndmap(pix2sky(wcs, pix, safe), wcs)

def pix2sky(wcs, pix, safe=True):
	"""Given an array of corner-based pixel coordinates [{y,x},...],
	return sky coordinates in the same ordering."""
	pix = np.asarray(pix)-0.5
	pflat = pix.reshape(pix.shape[0], np.prod(pix.shape[1:]))
	coords = np.asarray(wcs.wcs_pix2world(*(tuple(pflat)[::-1]+(0,)))[::-1])*np.pi/180
	coords = coords.reshape(pix.shape)
	if safe: coords = enlib.utils.unwind(coords)
	return coords

def sky2pix(wcs, coords, safe=True):
	"""Given an array of coordinates [{ra,dec},...], return
	pixel coordinates with the same ordering."""
	coords = np.asarray(coords)*180/np.pi
	cflat  = coords.reshape(coords.shape[0], np.prod(coords.shape[1:]))
	# period of the system
	pix = np.asarray(wcs.wcs_world2pix(*tuple(cflat)[::-1]+(0,)))+0.5
	if safe:
		for i in range(len(pix)):
			pix[i] = enlib.utils.unwind(pix[i], np.abs(360./wcs.wcs.cdelt[i]))
	return pix[::-1].reshape(coords.shape)

def project(map, shape, wcs, order=3):
	"""Project the map into a new map given by the specified
	shape and wcs, interpolating as necessary. Handles nan
	regions in the map by masking them before interpolating.
	This uses local interpolation, and will lose information
	when downgrading compared to averaging down."""
	map  = map.copy()
	pix  = map.sky2pix(posmap(shape, wcs))
	mask = ~np.isfinite(map)
	map[mask] = 0
	pmap = enlib.utils.interpol(map, pix, order=order)
	if np.sum(mask) > 0:
		pmask = np.abs(enlib.utils.interpol(1.0-mask, pix, order=min(1,order)))<1e-3
		pmap[pmask] = np.nan
	return ndmap(pmap, wcs)

############
# File I/O #
############

def write_map(fname, emap, fmt=None):
	"""Writes an enmap to file. If fmt is not passed,
	the file type is inferred from the file extension, and can
	be either fits or hdf. This can be overriden by
	passing fmt with either 'fits' or 'hdf' as argument."""
	if fmt == None:
		if   fname.endswith(".fits"): fmt = "fits"
		elif fname.endswith(".hdf"):  fmt = "hdf"
	if fmt == "fits":
		write_fits(fname, emap)
	elif fmt == "hdf":
		write_hdf(fname, emap)
	else:
		raise ValueError

def read_map(fname, fmt=None):
	"""Read an enmap from file. The file type is inferred
	from the file extension, unless fmt is passed.
	fmt must be one of 'fits' and 'hdf'."""
	toks = fname.split(":")
	fname = toks[0]
	if fmt == None:
		if   fname.endswith(".fits"): fmt = "fits"
		elif fname.endswith(".hdf"):  fmt = "hdf"
	if fmt == "fits":
		res = read_fits(fname)
	elif fmt == "hdf":
		res = read_hdf(fname)
	else:
		raise ValueError
	if len(toks) > 1:
		res = eval("res"+":".join(toks[1:]))
	return res

def write_fits(fname, emap):
	"""Write an enmap to a fits file."""
	# Get our basic wcs header
	header = emap.wcs.to_header()
	# Add our map headers
	header['NAXIS'] = emap.ndim
	for i,n in enumerate(emap.shape[:-1]):
		header['NAXIS%d'%(i+1)] = n
	hdus   = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(emap, header)])
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		hdus.writeto(fname, clobber=True)

def read_fits(fname, hdu=0):
	"""Read an enmap from the specified fits file. By default,
	the map and coordinate system will be read from HDU 0. Use
	the hdu argument to change this. The map must be stored as
	a fits image."""
	hdu = astropy.io.fits.open(fname)[hdu]
	with warnings.catch_warnings():
		wcs = enlib.wcs.WCS(hdu.header).sub(2)
	return ndmap(hdu.data, wcs)

def write_hdf(fname, emap):
	"""Write an enmap as an hdf file, preserving all
	the WCS metadata."""
	with h5py.File(fname, "w") as hfile:
		hfile["data"] = emap
		header = emap.wcs.to_header()
		for key in header:
			hfile["wcs/"+key] = header[key]

def read_hdf(fname):
	"""Read an enmap from the specified hdf file. Two formats
	are supported. The old enmap format, which simply used
	a bounding box to specify the coordinates, and the new
	format, which uses WCS properties. The latter is used if
	available. With the old format, plate carree projection
	is assumed. Note: some of the old files have a slightly
	buggy wcs, which can result in 1-pixel errors."""
	with h5py.File(fname,"r") as hfile:
		data = hfile["data"].value
		if "wcs" in hfile:
			hwcs = hfile["wcs"]
			header = astropy.io.fits.Header()
			for key in hwcs:
				header[key] = hwcs[key].value
			wcs = enlib.wcs.WCS(header).sub(2)
			return ndmap(data, wcs)
		else:
			# Compatibility for old format
			sys = hfile["system"].value if "system" in hfile else "equ"
			if sys == "equ": sys = "car"
			wcs = enlib.wcs.from_bounds(data.shape[-2:][::-1], hfile["box"].value[:,::-1]*np.pi/180, sys)
			return ndmap(data, wcs)
