import numpy as np, scipy.ndimage, warnings, enlib.utils, enlib.wcs, enlib.slice, enlib.fft, enlib.powspec, astropy.io.fits, sys
try:
	import h5py
except ImportError:
	pass
try:
	# Ensure that our numpy has stacked array support
	np.linalg.cholesky(np.array([1]).reshape(1,1,1))
	def svd_pow(a,e):
		U, s, Vh = np.linalg.svd(a)
		if e < 0: s[s==0] = np.inf
		seU = s[:,None,:]**e*U
		return np.einsum("xij,xjk->xik",seU,Vh)
except np.linalg.LinAlgError:
	try:
		from linmulti import svd_pow
	except ImportError:
		# This fallback is quite slow, even though the loop only has 10000 or so iterations
		def svd_pow(a, e):
			res = np.zeros(a.shape)
			for i in range(res.shape[0]):
				if not np.all(a[i]==0):
					U, s, Vh = np.linalg.svd(a[i])
					res[i] = (s**e*U).dot(Vh)
			return res

# Things that could be improved:
#  1. We assume exactly 2 WCS axes in spherical projection in {dec,ra} order.
#     It would be nice to support other configurations too. I have for example
#     needed [det,ra] or even [time,det,ra]. Adding support for this would
#     probably necessitate breaking backwards compatibility due to units.
#     WCS uses the units specified in the fits file, but I use radians.
#     Once we allos non-degree axes, the simple pi/180 conversion I use
#     won't work for all axes. It is simpler to just go with the flow and
#     use the same units as wcs. I need to think about how this would
#     interact with fourier units. Also, reordering or removing axes
#     can be difficult. I disallow that now, but for > 2 wcs dimensions,
#     these would be useful operations.
#  2. Passing around shape, wcs, dtype all the time is tedious. A simple
#     geometry object would make this less tedious, as long as it is
#     simple to override individual properties.

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
	def __str__(self): return repr(self)
	def __getitem__(self, sel):
		return np.ndarray.__getitem__(self, sel)
	def __array_wrap__(self, arr, context=None):
		if arr.ndim < 2: return arr
		return ndmap(arr, self.wcs)
	def copy(self, order='K'):
		return ndmap(np.copy(self,order), self.wcs)
	def sky2pix(self, coords, safe=True, corner=False): return sky2pix(self.shape, self.wcs, coords, safe, corner)
	def pix2sky(self, pix,    safe=True, corner=False): return pix2sky(self.shape, self.wcs, pix,    safe, corner)
	def box(self): return box(self.shape, self.wcs)
	def posmap(self, corner=False): return posmap(self.shape, self.wcs, corner=corner)
	def pixmap(self): return pixmap(self.shape, self.wcs)
	def lmap(self, oversample=1): return lmap(self.shape, self.wcs, oversample=oversample)
	def area(self): return area(self.shape, self.wcs)
	def extent(self): return extent(self.shape, self.wcs)
	@property
	def preflat(self):
		"""Returns a view of the map with the non-pixel dimensions flattened."""
		return self.reshape(-1, self.shape[-2], self.shape[-1])
	@property
	def npix(self): return np.product(self.shape[-2:])
	def project(self, shape, wcs, order=3, mode="nearest"): return project(self, shape, wcs, order, mode=mode, cval=0)
	def autocrop(self, method="plain", value="auto", margin=0, factors=None, return_info=False): return autocrop(self, method, value, margin, factors, return_info)
	def apod(self, width): return apod(self, width)
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
		if any([type(s) is not slice for s in sel2]):
			return np.asarray(self)[sel]
		# Otherwise we will return a full ndmap, including a
		# (possibly) sliced wcs.
		_, wcs = slice_wcs(self.shape[-2:], self.wcs, sel2)
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
		return self[...,ibox[0,0]:ibox[1,0]:ibox[2,0],ibox[0,1]:ibox[1,1]:ibox[2,1]]
	def subinds(self, box, inclusive=False):
		"""Helper function for submap. Translates the bounding
		box provided into a pixel units. Assumes rectangular
		coordinates."""
		box  = np.asarray(box)
		# Translate the box to pixels. The 0.5 moves us from
		# pixel-center coordinates to pixel-edge coordinates,
		# which we need to distinguish between fully or partially
		# included pixels
		bpix = self.wcs.wcs_world2pix(box[:,::-1]*180/np.pi,0)[:,::-1]+0.5
		dir  = 2*(bpix[1]>bpix[0])-1
		# If we are inclusive, find a bounding box, otherwise,
		# an internal box
		if inclusive:
			ibox = np.array([np.floor(bpix[0]),np.ceil(bpix[1]),dir],dtype=int)
		else:
			ibox = np.array([np.ceil(bpix[0]),np.floor(bpix[1]),dir],dtype=int)
		return ibox
	def write(self, fname, fmt=None):
		write_map(fname, self, fmt=fmt)

def slice_wcs(shape, wcs, sel):
	"""Slice a geometry specified by shape and wcs according to the
	slice sel. Returns a tuple of the output shape and the correponding
	wcs."""
	wcs = wcs.deepcopy()
	pre, shape = shape[:-2], shape[-2:]
	oshape = np.array(shape)
	# The wcs object has the indices in reverse order
	for i,s in enumerate(sel):
		s = enlib.slice.expand_slice(s, shape[i])
		j = -1-i
		start = s.start if s.step > 0 else s.start + 1
		wcs.wcs.crpix[j] -= start+0.5
		wcs.wcs.crpix[j] /= s.step
		wcs.wcs.cdelt[j] *= s.step
		wcs.wcs.crpix[j] += 0.5
		oshape[i] = s.stop-s.start
		oshape[i] = (oshape[i]+s.step-1)/s.step
	return tuple(pre)+tuple(oshape), wcs

def scale_wcs(wcs, factor):
	return enlib.wcs.scale(wcs, factor, rowmajor=True)

def box(shape, wcs, npoint=10):
	"""Compute a bounding box for the given geometry."""
	# Because of wcs's wrapping, we need to evaluate several
	# extra pixels to make our unwinding unambiguous
	pix = np.array([np.linspace(0,shape[-2],num=npoint,endpoint=True),
		np.linspace(0,shape[-1],num=npoint,endpoint=True)])-0.5
	coords = wcs.wcs_pix2world(pix[1],pix[0],0)[::-1]
	return enlib.utils.unwind(np.array(coords)*np.pi/180).T[[0,-1]]

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
		arr = np.asanyarray(arr, dtype=dtype).copy()
	if wcs is None:
		if isinstance(arr, ndmap):
			wcs = arr.wcs
		else:
			wcs = create_wcs(arr.shape)
	return ndmap(arr, wcs)

def empty(shape, wcs=None, dtype=None):
	return enmap(np.empty(shape, dtype=dtype), wcs, copy=False)
def zeros(shape, wcs=None, dtype=None):
	return enmap(np.zeros(shape, dtype=dtype), wcs, copy=False)
def ones(shape, wcs=None, dtype=None):
	return enmap(np.ones(shape, dtype=dtype), wcs, copy=False)
def full(shape, wcs, val, dtype=None):
	return enmap(np.full(shape, val, dtype=dtype), wcs, copy=False)

def posmap(shape, wcs, safe=True, corner=False):
	"""Return an enmap where each entry is the coordinate of that entry,
	such that posmap(shape,wcs)[{0,1},j,k] is the {y,x}-coordinate of
	pixel (j,k) in the map. Results are returned in radians, and
	if safe is true (default), then sharp coordinate edges will be
	avoided."""
	pix    = np.mgrid[:shape[-2],:shape[-1]]
	return ndmap(pix2sky(shape, wcs, pix, safe, corner), wcs)

def pixmap(shape, wcs=None):
	"""Return an enmap where each entry is the pixel coordinate of that entry."""
	res = np.mgrid[:shape[-2],:shape[-1]]
	return res if wcs is None else ndmap(res,wcs)

def pix2sky(shape, wcs, pix, safe=True, corner=False):
	"""Given an array of corner-based pixel coordinates [{y,x},...],
	return sky coordinates in the same ordering."""
	pix = np.asarray(pix).astype(float)
	if corner: pix -= 0.5
	pflat = pix.reshape(pix.shape[0], np.prod(pix.shape[1:]))
	coords = np.asarray(wcs.wcs_pix2world(*(tuple(pflat)[::-1]+(0,)))[::-1])*np.pi/180
	coords = coords.reshape(pix.shape)
	if safe: coords = enlib.utils.unwind(coords)
	return coords

def sky2pix(shape, wcs, coords, safe=True, corner=False):
	"""Given an array of coordinates [{dec,ra},...], return
	pixel coordinates with the same ordering. The corner argument
	specifies whether pixel coordinates start at pixel corners
	or pixel centers. This represents a shift of half a pixel.
	If corner is False, then the integer pixel closest to a position
	is round(sky2pix(...)). Otherwise, it is floor(sky2pix(...))."""
	coords = np.asarray(coords)*180/np.pi
	cflat  = coords.reshape(coords.shape[0], np.prod(coords.shape[1:]))
	# Quantities with a w prefix are in wcs ordering (ra,dec)
	wpix = np.asarray(wcs.wcs_world2pix(*tuple(cflat)[::-1]+(0,)))
	wshape = shape[-2:][::-1]
	if corner: wpix += 0.5
	if safe:
		# Put the angle cut as far away from the map as possible.
		# We do this by putting the reference point in the middle
		# of the map.
		wrefpix = np.array(wshape)/2
		if corner: wrefpix += 0.5
		for i in range(len(wpix)):
			wn = np.abs(360./wcs.wcs.cdelt[i])
			wpix[i] = enlib.utils.rewind(wpix[i], wrefpix[i], wn)
	return wpix[::-1].reshape(coords.shape)

def project(map, shape, wcs, order=3, mode="nearest", cval=0.0):
	"""Project the map into a new map given by the specified
	shape and wcs, interpolating as necessary. Handles nan
	regions in the map by masking them before interpolating.
	This uses local interpolation, and will lose information
	when downgrading compared to averaging down."""
	map  = map.copy()
	pix  = map.sky2pix(posmap(shape, wcs))
	pmap = enlib.utils.interpol(map, pix, order=order, mode=mode, cval=cval)
	return ndmap(pmap, wcs)

def rand_map(shape, wcs, cov, scalar=False):
	"""Generate a standard flat-sky pixel-space CMB map in TQU convention based on
	the provided power spectrum."""
	if scalar:
		return ifft(rand_gauss_iso_harm(shape, wcs, cov)).real
	else:
		return harm2map(rand_gauss_iso_harm(shape, wcs, cov))

def rand_gauss(shape, wcs, dtype=None):
	"""Generate a map with random gaussian noise in pixel space."""
	return ndmap(np.random.standard_normal(shape), wcs).astype(dtype,copy=False)

def rand_gauss_harm(shape, wcs):
	"""Mostly equivalent to np.fft.fft2(np.random.standard_normal(shape)),
	but avoids the fft by generating the numbers directly in frequency
	domain. Does not enforce the symmetry requried for a real map. If box is
	passed, the result will be an enmap."""
	return ndmap(np.random.standard_normal(shape)+1j*np.random.standard_normal(shape),wcs)

def rand_gauss_iso_harm(shape, wcs, cov):
	"""Generates an isotropic random map with component covariance
	cov in harmonic space, where cov is a (comp,comp,l) array."""
	data = map_mul(spec2flat(shape, wcs, cov, 0.5, mode="constant"), rand_gauss_harm(shape, wcs))
	return ndmap(data, wcs)

# Approximations to physical box size and area are needed
# for transforming to l-space. We can do this by dividing
# our map into a set of rectangles and computing the
# coordinates of their corners. The rectangles are assumed
# to be small, so cos(dec) is constant across them, letting
# us rescale RA by cos(dec) inside each. We also assume each
# rectangle to be .. a rectangle (:D), so area is given by
# two side lengths.
# The total length in each direction could be computed by
# 1. Average of top and bottom length
# 2. Mean of all row lengths
# 3. Area-weighted mean of row lengths
# 4. Some sort of compromise that makes length*height = area.
# To construct the coarser system, slicing won't do, as it
# shaves off some of our area. Instead, we must modify
# cdelt to match our new pixels: cdelt /= nnew/nold
def extent(shape, wcs, nsub=0x10):
	"""Returns an estimate of the "physical" extent of the
	patch given by shape and wcs as [height,width] in
	radians. That is, if the patch were on a sphere with
	radius 1 m, then this function returns approximately how many meters
	tall and wide the patch is. These are defined such that
	their product equals the physical area of the patch.
	Obs: Has trouble with areas near poles."""
	# Create a new wcs with (nsub,nsub) pixels
	wcs = wcs.deepcopy()
	step = (np.asfarray(shape[-2:])/nsub)[::-1]
	wcs.wcs.crpix -= 0.5
	wcs.wcs.cdelt *= step
	wcs.wcs.crpix /= step
	wcs.wcs.crpix += 0.5
	# Get position of all the corners, including the far ones
	pos = posmap([nsub+1,nsub+1], wcs, corner=True)
	# Apply az scaling
	scale = np.zeros([2,nsub,nsub])
	scale[1] = np.cos(0.5*(pos[0,1:,:-1]+pos[0,:-1,:-1]))
	scale[0] = 1
	ly = np.sum(((pos[:,1:,:-1]-pos[:,:-1,:-1])*scale)**2,0)**0.5
	lx = np.sum(((pos[:,:-1,1:]-pos[:,:-1,:-1])*scale)**2,0)**0.5
	areas = ly*lx
	# Compute approximate overall lengths
	Ay, Ax = np.sum(areas,0), np.sum(areas,1)
	Ly = np.sum(np.sum(ly,0)*Ay)/np.sum(Ay)
	Lx = np.sum(np.sum(lx,1)*Ax)/np.sum(Ax)
	return np.array([Ly,Lx])

def area(shape, wcs, nsub=0x10):
	"""Returns the area of a patch with the given shape
	and wcs, in steradians."""
	return np.prod(extent(shape, wcs, nsub))

def lmap(shape, wcs, oversample=1):
	"""Return a map of all the wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	ly, lx = laxes(shape, wcs, oversample=oversample)
	data = np.empty((2,ly.size,lx.size))
	data[0] = ly[:,None]
	data[1] = lx[None,:]
	return ndmap(data, wcs)

def laxes(shape, wcs, oversample=1):
	overample = int(oversample)
	step = extent(shape, wcs)/shape[-2:]
	ly = np.fft.fftfreq(shape[-2]*oversample, step[0])*2*np.pi
	lx = np.fft.fftfreq(shape[-1]*oversample, step[1])*2*np.pi
	if oversample > 1:
		# When oversampling, we want even coverage of fourier-space
		# pixels. Because the pixel value indicates the *center* l
		# of that pixel, we must shift ls when oversampling.
		# E.g. [0,100,200,...] oversample 2 => [-25,25,75,125,175,...],
		# not [0,50,100,150,200,...].
		# And  [0,100,200,...] os 3 => [-33,0,33,66,100,133,...]
		# In general [0,a,2a,3a,...] os n => a*(-1+(2*i+1)/n)/2
		# Since fftfreq generates a*i, the difference is a/2*(-1+1/n)
		def shift(l,a,n): return l+a/2*(-1+1./n)
		ly = shift(ly,ly[oversample],oversample)
		lx = shift(lx,lx[oversample],oversample)
	return ly, lx

def lrmap(shape, wcs, oversample=1):
	"""Return a map of all the wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	return lmap(shape, wcs, oversample=oversample)[...,:shape[-1]/2+1]

def fft(emap, omap=None, nthread=0, normalize=True):
	"""Performs the 2d FFT of the enmap pixels, returning a complex enmap."""
	res = samewcs(enlib.fft.fft(emap,omap,axes=[-2,-1],nthread=nthread), emap)
	if normalize: res /= np.prod(emap.shape[-2:])**0.5
	return res
def ifft(emap, omap=None, nthread=0, normalize=True):
	"""Performs the 2d iFFT of the complex enmap given, and returns a pixel-space enmap."""
	res = samewcs(enlib.fft.ifft(emap,omap,axes=[-2,-1],nthread=nthread, normalize=False), emap)
	if normalize: res /= np.prod(emap.shape[-2:])**0.5
	return res

# These are shortcuts for transforming from T,Q,U real-space maps to
# T,E,B hamonic maps. They are not the most efficient way of doing this.
# It would be better to precompute the rotation matrix and buffers, and
# use real transforms.
def map2harm(emap, nthread=0):
	"""Performs the 2d FFT of the enmap pixels, returning a complex enmap."""
	emap = samewcs(fft(emap,nthread=nthread), emap)
	if emap.ndim > 2 and emap.shape[-3] > 1:
		rot = queb_rotmat(emap.lmap())
		emap[...,-2:,:,:] = map_mul(rot, emap[...,-2:,:,:])
	return emap
def harm2map(emap, nthread=0, normalize=True):
	if emap.ndim > 2 and emap.shape[-3] > 1:
		rot = queb_rotmat(emap.lmap(), inverse=True)
		emap = emap.copy()
		emap[...,-2:,:,:] = map_mul(rot, emap[...,-2:,:,:])
	return samewcs(ifft(emap,nthread=nthread), emap).real

def queb_rotmat(lmap, inverse=False):
	a    = 2*np.arctan2(lmap[0], lmap[1])
	c, s = np.cos(a), np.sin(a)
	if inverse: s = -s
	return samewcs(np.array([[c,-s],[s,c]]),lmap)

def rotate_pol(emap, angle, comps=[-2,-1]):
	c, s = np.cos(2*angle), np.sin(2*angle)
	res = emap.copy()
	res[...,comps[0],:,:] = c*emap[...,comps[0],:,:] - s*emap[...,comps[1],:,:]
	res[...,comps[1],:,:] = s*emap[...,comps[0],:,:] + c*emap[...,comps[1],:,:]
	return res

def map_mul(mat, vec):
	"""Elementwise matrix multiplication mat*vec. Result will have
	the same shape as vec. Multiplication happens along the first indices.
	This function is buggy when mat is not square (in the multiplication
	dimensions). This is due to the reshape at the end. I should figure out
	what code depends on that, and decide what I really want this function
	to do."""
	oshape= vec.shape
	if len(oshape) == 2: oshape = (1,)+oshape
	tvec = np.reshape(vec, oshape)
	# It is a bit clunky to get einsum to handle arbitrary numbers of dimensions.
	vpre  = "".join([chr(ord('a')+i) for i in range(len(oshape)-3)])
	mpre  = vpre[vec.ndim-(mat.ndim-1):]
	data  = np.reshape(np.einsum("%sxyzw,%syzw->%sxzw" % (mpre,vpre,vpre), mat, tvec), vec.shape)
	return samewcs(data, mat, vec)

def smooth_gauss(emap, sigma):
	"""Smooth the map given as the first argument with a gaussian beam
	with the given standard deviation in radians."""
	if sigma == 0: return emap.copy()
	f  = map2harm(emap)
	l2 = np.sum(emap.lmap()**2,0)
	f *= np.exp(-l2*sigma**2)
	return harm2map(f)

def calc_window(shape):
	"""Compute fourier-space window function. Like the other fourier-based
	functions in this module, equi-spaced pixels are assumed. Since the
	window function is separable, it is returned as an x and y part,
	such that window = wy[:,None]*wx[None,:]."""
	wy = np.sinc(np.fft.fftfreq(shape[-2]))
	wx = np.sinc(np.fft.fftfreq(shape[-1]))
	return wy, wx

def apply_window(emap, pow=1.0):
	"""Apply the pixel window function to the specified power to the map,
	returning a modified copy. Use pow=-1 to unapply the pixel window."""
	wy, wx = calc_window(emap.shape)
	return ifft(fft(emap) * wy[:,None]**pow * wx[None,:]**pow).real

def samewcs(arr, *args):
	"""Returns arr with the same wcs information as the first enmap among args.
	If no mathces are found, arr is returned as is."""
	for m in args:
		try: return ndmap(arr, m.wcs)
		except AttributeError: pass
	return arr

# Idea: Make geometry a class with .shape and .wcs members.
# Make a function that takes (foo,bar) and returns a geometry,
# there (foo,bar) can either be (shape,wcs) or (geometry,None).
# Use that to make everything that currently accepts shape, wcs
# transparently accept geometry. This will free us from having
# to drag around a shape, wcs pair all the time.
def geometry(pos, res=None, shape=None, proj="cea", deg=False, pre=(), **kwargs):
	"""Consruct a shape,wcs pair suitable for initializing enmaps.
	pos can be either a [2] center position or a [{from,to},2]
	bounding box. At least one of res or shape must be specified.
	If res is specified, it must either be a number, in
	which the same resolution is used in each direction,
	or [2]. If shape is specified, it must be [2]. All angles
	are given in radians."""
	scale = 1 if deg else 180/np.pi
	pos = np.asarray(pos)*scale
	if res is not None: res = np.asarray(res)*scale
	wcs = enlib.wcs.build(pos, res, shape, rowmajor=True, system=proj, **kwargs)
	if shape is None:
		# Infer shape. WCS does not allow us to wrap around the
		# sky, so shape mustn't be large enough to make that happen.
		# Our relevant pixel coordinates go from (-0.5,-0.5) to
		# shape-(0.5,0.5). We assume that wcs.build has already
		# assured the former. Our job is to find shape that puts
		# the top edge close to the requested value, while still
		# being valied. If we always round down, we should be safe:
		faredge = wcs.wcs_world2pix(pos[1:2,::-1],0)[0,::-1]
		shape = tuple(np.floor(faredge+0.5).astype(int))
	return pre+tuple(shape), wcs


#class geometry:
#	def __init__(self, shape=None, wcs=None, pos=None, res=None, proj=None, rad=True, **kwargs):
#		"""Construct a new geometry using one of several formats:
#		1. geometry(geometry)
#		2. geometry(shape, wcs)
#		3. geometry(pos=box, res=res)
#		4. geometry(shape, pos=box)
#		5. geometry(shape, pos=center, res=res)
#		6. geometry() (constructs a 10x10 degree area centered on 0,0 with 1 arcmin res)
#		box is [[dec_from,ra_from],[dec_to,ra_to]]
#		center is [dec_center,ra_center]
#		res = num or [res_dec,res_ra]
#		All numbers are in radians unless rad=False is specdified, in which case
#		pos is in degrees and res in minutes of arc."""
#		if isinstance(shape, geometry):
#			self.shape, self.wcs = tuple(shape.shape), shape.wcs.copy()
#		elif shape is not None and wcs is not None:
#			self.shape, self.wcs = tuple(shape), wcs.copy()
#		else:
#			pconv = 180/np.pi if args.rad else 1.0
#			rconv = 180/np.pi if args.rad else 1.0/60
#			if pos is None:
#				pos = np.array([[-5,-5],[5,5]])
#			else:
#				pos = np.asarray(pos)*pconv
#			if res is None and shape is None: res = 1.0/60
#			if res is not None: res = np.asarray(res)*rconv
#			if proj is None: proj = "cea"
#			wcs = enlib.wcs.build(pos, res, shape, rowmajor=True, system=proj, **kwargs)
#			if shape is None:
#				# Infer shape
#				corners = wcs.wcs_world2pix(pos[:,::-1],1)
#				shape = tuple(np.ceil(np.abs(corners[1]-corners[0])).astype(int))[::-1]
#			self.shape, self.wcs = tuple(shape), wcs
#	def box(self, npoint=10):
#		"""Compute a bounding box for the given geometry."""
#		# Because of wcs's wrapping, we need to evaluate several
#		# extra pixels to make our unwinding unambiguous
#		pix = np.array([np.linspace(0,self.shape[-2],num=npoint,endpoint=True),
#			np.linspace(0,self.shape[-1],num=npoint,endpoint=True)])-0.5
#		coords = self.wcs.wcs_pix2world(pix[1],pix[0],1)[::-1]
#		return enlib.utils.unwind(np.array(coords)*np.pi/180).T[[0,-1]]
#	# Approximations to physical box size and area are needed
#	# for transforming to l-space. We can do this by dividing
#	# our map into a set of rectangles and computing the
#	# coordinates of their corners. The rectangles are assumed
#	# to be small, so cos(dec) is constant across them, letting
#	# us rescale RA by cos(dec) inside each. We also assume each
#	# rectangle to be .. a rectangle (:D), so area is given by
#	# two side lengths.
#	# The total length in each direction could be computed by
#	# 1. Average of top and bottom length
#	# 2. Mean of all row lengths
#	# 3. Area-weighted mean of row lengths
#	# 4. Some sort of compromise that makes length*height = area.
#	# To construct the coarser system, slicing won't do, as it
#	# shaves off some of our area. Instead, we must modify
#	# cdelt to match our new pixels: cdelt /= nnew/nold
#	def extent(self, nsub=0x10):
#		"""Returns an estimate of the "physical" extent of the
#		patch given by this geometry as [height,width] in
#		radians. That is, if the patch were on a sphere with
#		radius 1 m, then this function returns approximately how many meters
#		tall and width the patch is. These are defined such that
#		their product equals the physical area of the patch."""
#		wcs = self.wcs.copy()
#		step = (np.asfarray(self.shape[-2:])/nsub)[::-1]
#		wcs.wcs.cdelt *= step
#		wcs.wcs.crpix /= step
#		# Get position of all the corners, including the far ones
#		pos = posmap([nsub+1,nsub+1], wcs, corner=True)
#		# Apply az scaling
#		scale = np.zeros([2,nsub,nsub])
#		scale[0] = np.cos(0.5*(pos[0,1:,:-1]+pos[0,:-1,:-1]))
#		scale[1] = 1
#		ly = np.sum(((pos[:,1:,:-1]-pos[:,:-1,:-1])*scale)**2,0)**0.5
#		lx = np.sum(((pos[:,:-1,1:]-pos[:,:-1,:-1])*scale)**2,0)**0.5
#		areas = ly*lx
#		# Compute approximate overall lengths
#		Ay, Ax = np.sum(areas,0), np.sum(areas,1)
#		Ly = np.sum(np.sum(ly,0)*Ay)/np.sum(Ay)
#		Lx = np.sum(np.sum(lx,1)*Ax)/np.sum(Ax)
#		return np.array([Ly,Lx])
#	def area(self, nsub=0x10):
#		"""Returns the area of a patch with this geometry, in steradians."""
#		return np.prod(self.extent(nsub=nsub))
#	def pix2sky(self, pix, safe=True, corner=False):
#		"""Given an array of corner-based pixel coordinates [{y,x},...],
#		return sky coordinates in the same ordering."""
#		pix = np.asarray(pix).astype(float)
#		if corner: pix -= 0.5
#		pflat = pix.reshape(pix.shape[0], np.prod(pix.shape[1:]))
#		coords = np.asarray(self.wcs.wcs_pix2world(*(tuple(pflat)[::-1]+(1,)))[::-1])*np.pi/180
#		coords = coords.reshape(pix.shape)
#		if safe: coords = enlib.utils.unwind(coords)
#		return coords
#	def sky2pix(self, coords, safe=True, corner=False):
#		"""Given an array of coordinates [{ra,dec},...], return
#		pixel coordinates with the same ordering. The corner argument
#		specifies whether pixel coordinates start at pixel corners
#		or pixel centers. This represents a shift of half a pixel.
#		If corner is False, then the integer pixel closest to a position
#		is round(sky2pix(...)). Otherwise, it is floor(sky2pix(...))."""
#		coords = np.asarray(coords)*180/np.pi
#		cflat  = coords.reshape(coords.shape[0], np.prod(coords.shape[1:]))
#		# period of the system
#		pix = np.asarray(self.wcs.wcs_world2pix(*tuple(cflat)[::-1]+(1,)))
#		if corner: pix += 0.5
#		if safe:
#			# Put the angle cut as far away from the map as possible.
#			# We do this by putting the reference point in the middle
#			# of the map.
#			refpix = np.array(self.shape[-2:])/2
#			if corner: refpix += 0.5
#			for i in range(len(pix)):
#				n = np.abs(360./self.wcs.wcs.cdelt[i])
#				pix[i] = enlib.utils.rewind(pix[i], refpix[i], n)
#		return pix[::-1].reshape(coords.shape)
#
#def gwrap(shape=None, wcs=None):
#	"""This function transparently accepts either a shape,wcs pair,
#	both of which may be None in which case they remain None as the output,
#	or a geometry object which is unpacked into a shape and wcs.
#	The result is always a shape,wcs tuple."""
#	try:
#		return shape.shape, shape.wcs
#	except AttributeError:
#		return shape, wcs

def create_wcs(shape, box=None, proj="cea"):
	if box is None: box = np.array([[-1,-1],[1,1]])*0.5*10*np.pi/180
	return enlib.wcs.build(box*180/np.pi, shape=shape, rowmajor=True, system=proj)

def spec2flat(shape, wcs, cov, exp=1.0, mode="nearest", oversample=1, smooth="auto"):
	"""Given a (ncomp,ncomp,l) power spectrum, expand it to harmonic map space,
	returning (ncomp,ncomp,y,x). This involves a rescaling which converts from
	power in terms of multipoles, to power in terms of 2d frequency.
	The optional exp argument controls the exponent of the rescaling factor.
	To use this with the inverse power spectrum, pass exp=-1, for example.
	If apply_exp is True, the power spectrum will be taken to the exp'th
	power. Otherwise, it is assumed that this has already been done, and
	the exp argument only controls the normalization of the result.

	It is irritating that this function needs to know what kind of matrix
	it is expanding, but I can't see a way to avoid it. Changing the
	units of harmonic space is not sufficient, as the following demonstrates:
	  m = harm2map(map_mul(spec2flat(s, b, multi_pow(ps, 0.5), 0.5), map2harm(rand_gauss(s,b))))
	The map m is independent of the units of harmonic space, and will be wrong unless
	the spectrum is properly scaled. Since this scaling depends on the shape of
	the map, this is the appropriate place to do so, ugly as it is."""
	oshape= tuple(shape)
	if len(oshape) == 2: oshape = (1,)+oshape
	ls = np.sum(lmap(oshape, wcs, oversample=oversample)**2,0)**0.5
	if smooth == "auto":
		# Determine appropriate fourier-scale smoothing based on 2d fourer
		# space resolution. We wish to smooth by about this width to approximate
		# averaging over sub-grid modes
		smooth = 0.5*(ls[1,0]+ls[0,1])
		smooth /= 3.41 # 3.41 is an empirical factor
	if smooth > 0:
		cov = smooth_spectrum(cov, kernel="gauss", weight="mode", width=smooth)
	# Translate from steradians to pixels
	cov = cov * np.prod(shape[-2:])/area(shape,wcs)
	if exp != 1.0: cov = multi_pow(cov, exp)
	cov[~np.isfinite(cov)] = 0
	cov   = cov[:oshape[-3],:oshape[-3]]
	# Use order 1 because we will perform very short interpolation, and to avoid negative
	# values in spectra that must be positive (and it's faster)
	res = ndmap(enlib.utils.interpol(cov, np.reshape(ls,(1,)+ls.shape),mode=mode, mask_nan=False, order=1),wcs)
	res = downgrade(res, oversample)
	return res

def smooth_spectrum(ps, kernel="gauss", weight="mode", width=1.0):
	"""Smooth the spectrum ps with the given kernel, using the given weighting."""
	ps = np.asanyarray(ps)
	pflat = ps.reshape(-1,ps.shape[-1])
	nspec,nl = pflat.shape
	# Set up the kernel array
	K = np.zeros((nspec,nl))
	l = np.arange(nl)
	if isinstance(kernel, basestring):
		if kernel == "gauss":
			K[:] = np.exp(-0.5*(l/width)**2)
		elif kernel == "step":
			K[:,:int(width)] = 1
		else:
			raise ValueError("Unknown kernel type %s in smooth_spectrum" % kernel)
	else:
		tmp = np.atleast_2d(kernel)
		K[:,:tmp.shape[-1]] = tmp[:,:K.shape[-1]]
	# Set up the weighting scheme
	W = np.zeros((nspec,nl))
	if isinstance(weight, basestring):
		if weight == "mode":
			W[:] = l[None,:]**2
		elif weight == "uniform":
			W[:] = 1
		else:
			raise ValueError("Unknown weighting scheme %s in smooth_spectrum" % weight)
	else:
		tmp = np.atleast_2d(weight)
		assert tmp.shape[-1] == W.shape[-1], "Spectrum weight must have the same length as spectrum"
	pWK = _convolute_sym(pflat*W, K)
	WK  = _convolute_sym(W, K)
	res = pWK/WK
	return res.reshape(ps.shape)

def _convolute_sym(a,b):
	sa = np.concatenate([a,a[:,-2:0:-1]],-1)
	sb = np.concatenate([b,b[:,-2:0:-1]],-1)
	fa = enlib.fft.rfft(sa)
	fb = enlib.fft.rfft(sb)
	sa = enlib.fft.ifft(fa*fb,sa,normalize=True)
	return sa[:,:a.shape[-1]]

def multi_pow(mat, exp, axes=[0,1]):
	"""Raise each sub-matrix of mat (ncomp,ncomp,...) to
	the given exponent in eigen-space."""
	res = enlib.utils.partial_expand(svd_pow(enlib.utils.partial_flatten(mat, axes, 0), exp), mat.shape, axes, 0)
	return samewcs(res, mat)

def downgrade(emap, factor):
	"""Returns enmap "emap" downgraded by the given integer factor
	(may be a list for each direction, or just a number) by averaging
	inside pixels."""
	fact = np.full(2, 1).astype(int)
	fact[:] = factor
	tshape = emap.shape[-2:]/fact*fact
	res = np.mean(np.reshape(emap[...,:tshape[0],:tshape[1]],emap.shape[:-2]+(tshape[0]/fact[0],fact[0],tshape[1]/fact[1],fact[1])),(-3,-1))
	return ndmap(res, emap[...,::fact[0],::fact[1]].wcs)

def upgrade(emap, factor):
	"""Upgrade emap to a larger size using nearest neighbor interpolation,
	returning the result. More advanced interpolation can be had using
	enmap.interpolate."""
	fact = np.full(2,1).astype(int)
	fact[:] = factor
	res = np.tile(emap.copy().reshape(emap.shape[:-2]+(emap.shape[-2],1,emap.shape[-1],1)),(1,fact[0],1,fact[1]))
	res = res.reshape(res.shape[:-4]+(np.product(res.shape[-4:-2]),np.product(res.shape[-2:])))
	# Correct the WCS information
	for j in range(2):
		res.wcs.wcs.crpix[j] -= 0.5
		res.wcs.wcs.crpix[j] *= fact[1-j]
		res.wcs.wcs.cdelt[j] /= fact[1-j]
		res.wcs.wcs.crpix[j] += 0.5
	return res

def pad(emap, pix, return_slice=False,wrap=False):
	"""Pad enmap "emap", creating a larger map with zeros filled in on the sides.
	How much to pad is controlled via pix. If pix is a scalar, it specifies the number
	of pixels to add on all sides. If it is 1d, it specifies the number of pixels to add
	at each end for each axis. If it is 2d, the number of pixels to add at each end
	of an axis can be specified individually."""
	pix = np.asarray(pix,dtype=int)
	if pix.ndim == 0:
		pix = np.array([[pix,pix],[pix,pix]])
	elif pix.ndim == 1:
		pix = np.array([pix,pix])
	# Exdend the wcs in each direction.
	w = emap.wcs.deepcopy()
	w.wcs.crpix += pix[0,::-1]
	# Construct a slice between the new and old map
	res = zeros(emap.shape[:-2]+tuple([s+sum(p) for s,p in zip(emap.shape[-2:],pix.T)]),wcs=w, dtype=emap.dtype)
	mslice = (Ellipsis,slice(pix[0,0],res.shape[-2]-pix[1,0]),slice(pix[0,1],res.shape[-1]-pix[1,1]))
	res[mslice] = emap
	if wrap:
		res[...,:pix[0,0],:]  = res[...,-pix[0,0]-pix[1,0]:-pix[1,0],:]
		res[...,-pix[1,0]:,:] = res[...,pix[0,0]:pix[0,0]+pix[1,0],:]
		res[...,:,:pix[0,1]]  = res[...,:,-pix[0,1]-pix[1,1]:-pix[1,1]]
		res[...,:,-pix[1,1]:] = res[...,:,pix[0,1]:pix[0,1]+pix[1,1]]
	return (res,mslice) if return_slice else res

def autocrop(m, method="plain", value="auto", margin=0, factors=None, return_info=False):
	"""Adjust the size of m to be more fft-friendly. If possible,
	blank areas at the edge of the map are cropped to bring us to a nice
	length. If there there aren't enough blank areas, the map is padded
	instead."""
	def calc_blanks(m, value):
		value   = np.asarray(value)
		hitmask = np.all(np.isclose(m.T, value.T, equal_nan=True, rtol=1e-6, atol=0).T,axis=tuple(range(m.ndim-2)))
		hitrows = np.all(hitmask,1)
		hitcols = np.all(hitmask,0)
		blanks  = np.array([np.where(~hitrows)[0][[0,-1]],np.where(~hitcols)[0][[0,-1]]]).T
		blanks[1] = m.shape[-2:]-blanks[1]-1
		return blanks
	if value == "auto":
		bs = [calc_blanks(m, m[...,i,j]) for  i in [0,-1] for j in [0,-1]]
		nb = [np.product(np.sum(b,0)) for b in bs]
		blanks = bs[np.argmax(nb)]
	else:
		blanks = calc_blanks(m, value)
	nblank  = np.sum(blanks,0)
	# Find the first good sizes larger than the unblank lengths
	minshape  = m.shape[-2:]-nblank+margin
	if method == "plain":
		goodshape = minshape
	elif method == "fft":
		goodshape = np.array([enlib.fft.fft_len(l, direction="above", factors=None) for l in minshape])
	else:
		raise ValueError("Unknown autocrop method %s!" % method)
	# Pad if necessary
	adiff   = np.maximum(0,goodshape-m.shape[-2:])
	padding = [[0,0],[0,0]]
	if any(adiff>0):
		padding = [adiff,[0,0]]
		m = pad(m, padding)
		blanks[0] += adiff
		nblank = np.sum(blanks,0)
	# Then crop to goodshape
	tocrop = m.shape[-2:]-goodshape
	lower  = np.minimum(tocrop,blanks[0])
	upper  = tocrop-lower
	s      = (Ellipsis,slice(lower[0],m.shape[-2]-upper[0]),slice(lower[1],m.shape[-1]-upper[1]))
	class PadcropInfo:
		slice   = s
		pad     = padding
	if return_info:
		return m[s], PadcropInfo
	else:
		return m[s]

def padcrop(m, info):
	return pad(m, info.pad)[info.slice]

def grad(m):
	"""Returns the gradient of the map m as [2,...]."""
	return ifft(fft(m)*_widen(m.lmap(),m.ndim+1)*1j).real

def grad_pix(m):
	"""The gradient of map m expressed in units of pixels.
	Not the same as the gradient of m with resepect to pixels.
	Useful for avoiding sky2pix-calls for e.g. lensing,
	and removes the complication of axes that increase in
	nonstandard directions."""
	return grad(m)*(m.shape[-2:]/m.extent())[(slice(None),)+(None,)*m.ndim]

def div(m):
	"""Returns the divergence of the map m[2,...] as [...]."""
	return ifft(np.sum(fft(m)*_widen(m.lmap(),m.ndim)*1j,0)).real

def _widen(map,n):
	"""Helper for gard and div. Adds degenerate axes between the first
	and the last two to give the map a total dimensionality of n."""
	return map[(slice(None),) + (None,)*(n-3) + (slice(None),slice(None))]

def apod(m, width, profile="cos"):
	width = np.minimum(np.zeros(2)+width,m.shape[-2:])
	if profile == "cos":
		a = [0.5*(1-np.linspace(0,np.pi,w)) for w in width]
	else:
		raise ValueError("Unknown apodization profile %s" % profile)
	res = m.copy()
	if width[0] > 0:
		res[...,:width[0],:] *= a[0][:,None]
		res[...,-width[0]:,:] *= a[0][::-1,None]
	if width[1] > 0:
		res[...,:,:width[1]] *= a[1][None,:]
		res[...,:,-width[1]:]  *= a[1][None,::-1]
	return res

def radial_average(map, center=[0,0], step=1.0):
	"""Produce a radial average of the given map that's centered on zero"""
	center = np.asarray(center)
	pos  = map.posmap()-center[:,None,None]
	rads = np.sum(pos**2,0)**0.5
	# Our resolution should be step times the highest resolution direction.
	res = np.min(map.extent()/map.shape[-2:])*step
	n   = int(np.max(rads/res))
	orads = np.arange(n)*res
	rinds = (rads/res).reshape(-1).astype(int)
	# Ok, rebin the map. We use this using bincount, which can be a bit slow
	mflat = map.reshape((-1,)+map.shape[-2:])
	mout = np.zeros((len(mflat),n))
	for i, m in enumerate(mflat):
		mout[i] = (np.bincount(rinds, weights=m.reshape(-1))/np.bincount(rinds))[:n]
	mout = mout.reshape(map.shape[:-2]+mout.shape[1:])
	return mout, orads

############
# File I/O #
############

def write_map(fname, emap, fmt=None):
	"""Writes an enmap to file. If fmt is not passed,
	the file type is inferred from the file extension, and can
	be either fits or hdf. This can be overriden by
	passing fmt with either 'fits' or 'hdf' as argument."""
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits"
		else: fmt = "fits"
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
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits"
		else: fmt = "fits"
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
	# The fits write routines may attempt to modify
	# the map. So make a copy.
	emap = emap.copy()
	# Get our basic wcs header
	header = emap.wcs.to_header()
	# Add our map headers
	header['NAXIS'] = emap.ndim
	for i,n in enumerate(emap.shape[::-1]):
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
	res = ndmap(hdu.data, wcs)
	if res.dtype.byteorder not in ['=','<' if sys.byteorder == 'little' else '>']:
		res = res.byteswap().newbyteorder()
	return res

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
			res = ndmap(data, wcs)
		else:
			# Compatibility for old format
			csys = hfile["system"].value if "system" in hfile else "equ"
			if csys == "equ": csys = "car"
			wcs = enlib.wcs.build(hfile["box"].value, shape=data.shape, system=csys, rowmajor=True)
			res = ndmap(data, wcs)
	if res.dtype.byteorder not in ['=','<' if sys.byteorder == 'little' else '>']:
		res = res.byteswap().newbyteorder()
	return res
