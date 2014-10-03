import numpy as np, scipy.ndimage, warnings, enlib.utils, enlib.wcs, enlib.slice, enlib.fft, astropy.io.fits, sys
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
	def copy(self, order='K'):
		return ndmap(np.copy(self,order), self.wcs)
	def sky2pix(self, coords, safe=True, corner=False): return sky2pix(self.wcs, coords, safe, corner)
	def pix2sky(self, pix,    safe=True, corner=False): return pix2sky(self.wcs, pix,    safe, corner)
	def box(self): return box(self.shape, self.wcs)
	def posmap(self, corner=False): return posmap(self.shape, self.wcs, corner=corner)
	def pixmap(self): return pixmap(self.shape, self.wcs)
	def lmap(self): return lmap(self.shape, self.wcs)
	def area(self): return area(self.shape, self.wcs)
	def extent(self): return extent(self.shape, self.wcs)
	def project(self, shape, wcs, order=3, mode="nearest"): return project(self, shape, wcs, order, mode=mode)
	def autocrop(self, method="plain", value=np.nan, margin=0, factors=None, return_info=False): return autocrop(self, method, value, margin, factors, return_info)
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

def slice_wcs(shape, wcs, sel):
	wcs = wcs.deepcopy()
	oshape = np.array(shape)
	# The wcs object has the indices in reverse order
	for i,s in enumerate(sel):
		s = enlib.slice.expand_slice(s, shape[i])
		j = -1-i
		wcs.wcs.crpix[j] -= s.start+0.5
		wcs.wcs.crpix[j] /= s.step
		wcs.wcs.cdelt[j] *= s.step
		wcs.wcs.crpix[j] += 0.5
		oshape[i] = (oshape[i]+s.step-1)/s.step
	return tuple(oshape), wcs

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
			wcs = create_wcs(arr.shape)
	return ndmap(arr, wcs)

def zeros(shape, wcs=None, dtype=None):
	return enmap(np.zeros(shape), wcs, dtype=dtype)
def empty(shape, wcs=None, dtype=None):
	return enmap(np.empty(shape), wcs, dtype=dtype)

def posmap(shape, wcs, safe=True, corner=False):
	"""Return an enmap where each entry is the coordinate of that entry,
	such that posmap(shape,wcs)[{0,1},j,k] is the {y,x}-coordinate of
	pixel (j,k) in the map. Results are returned in radians, and
	if safe is true (default), then sharp coordinate edges will be
	avoided."""
	pix    = np.mgrid[:shape[-2],:shape[-1]]
	return ndmap(pix2sky(wcs, pix, safe, corner), wcs)

def pixmap(shape, wcs=None):
	"""Return an enmap where each entry is the pixel coordinate of that entry."""
	res = np.mgrid[:shape[-2],:shape[-1]]
	return res if wcs is None else ndmap(res,wcs)

def pix2sky(wcs, pix, safe=True, corner=False):
	"""Given an array of corner-based pixel coordinates [{y,x},...],
	return sky coordinates in the same ordering."""
	pix = np.asarray(pix)
	if corner: pix -= 0.5
	pflat = pix.reshape(pix.shape[0], np.prod(pix.shape[1:]))
	coords = np.asarray(wcs.wcs_pix2world(*(tuple(pflat)[::-1]+(0,)))[::-1])*np.pi/180
	coords = coords.reshape(pix.shape)
	if safe: coords = enlib.utils.unwind(coords)
	return coords

def sky2pix(wcs, coords, safe=True, corner=False):
	"""Given an array of coordinates [{ra,dec},...], return
	pixel coordinates with the same ordering. The corner argument
	specifies whether pixel coordinates start at pixel corners
	or pixel centers. This represents a shift of half a pixel.
	If corner is False, then the integer pixel closest to a position
	is round(sky2pix(...)). Otherwise, it is floor(sky2pix(...))."""
	coords = np.asarray(coords)*180/np.pi
	cflat  = coords.reshape(coords.shape[0], np.prod(coords.shape[1:]))
	# period of the system
	pix = np.asarray(wcs.wcs_world2pix(*tuple(cflat)[::-1]+(0,)))
	if corner: pix += 0.5
	if safe:
		# Avoid angle cuts inside the map. Place the cut at the
		# opposite side of the sky relative to the zero pixel
		for i in range(len(pix)):
			n = np.abs(360./wcs.wcs.cdelt[i])
			pix[i] = enlib.utils.rewind(pix[i], pix[i,0]%n, n)
	return pix[::-1].reshape(coords.shape)

def project(map, shape, wcs, order=3, mode="nearest"):
	"""Project the map into a new map given by the specified
	shape and wcs, interpolating as necessary. Handles nan
	regions in the map by masking them before interpolating.
	This uses local interpolation, and will lose information
	when downgrading compared to averaging down."""
	map  = map.copy()
	pix  = map.sky2pix(posmap(shape, wcs), corner=True)
	pmap = enlib.utils.interpol(map, pix, order=order, mode=mode)
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
	data = map_mul(spec2flat(shape, wcs, cov, 0.5), rand_gauss_harm(shape, wcs))
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
	tall and width the patch is."""
	wcs = wcs.deepcopy()
	step = np.asfarray(shape[-2:])/nsub
	wcs.wcs.cdelt *= step
	wcs.wcs.crpix /= step
	# Get position of all the corners, including the far ones
	pos = posmap([nsub+1,nsub+1], wcs, corner=True)
	# Apply az scaling
	scale = np.zeros([2,nsub,nsub])
	scale[0] = np.cos(0.5*(pos[0,1:,:-1]+pos[0,:-1,:-1]))
	scale[1] = 1
	ly = np.sum(((pos[:,1:,:-1]-pos[:,:-1,:-1])*scale)**2,0)**0.5
	lx = np.sum(((pos[:,:-1,1:]-pos[:,:-1,:-1])*scale)**2,0)**0.5
	areas = ly*lx
	# Compute approximate overall lengths
	Ay, Ax = np.sum(areas,0), np.sum(areas,1)
	Ly = np.sum(np.sum(ly,0)*Ay)/np.sum(Ay)
	Lx = np.sum(np.sum(lx,1)*Ax)/np.sum(Ax)
	return np.array([Lx,Ly])

def area(shape, wcs, nsub=0x10):
	"""Returns the area of a patch with the given shape
	and wcs, in steradians."""
	return np.prod(extent(shape, wcs, nsub))

def lmap(shape, wcs):
	"""Return a map of all the wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	step = extent(shape, wcs)/shape[-2:]
	data = np.empty([2]+list(shape[-2:]))
	data[0] = np.fft.fftfreq(shape[-2], step[0])[:,None]
	data[1] = np.fft.fftfreq(shape[-1], step[1])[None,:]
	return ndmap(data, wcs)*2*np.pi

def lrmap(shape, wcs):
	"""Return a map of all the wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	return lmap(shape, wcs)[...,:shape[-1]/2+1]

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
	the same shape as vec. Multiplication happens along the first indices."""
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
	f  = map2harm(emap)
	l2 = np.sum(emap.lmap()**2,0)
	f *= np.exp(-l2*sigma**2)
	return harm2map(f)

def samewcs(arr, *args):
	"""Returns arr with the same wcs information as the first enmap among args.
	If no mathces are found, arr is returned as is."""
	for m in args:
		try: return ndmap(arr, m.wcs)
		except AttributeError: pass
	return arr

def create_wcs(shape, box=None, proj="cea"):
	if box is None: box = np.array([[-1,-1],[1,1]])*0.5*10*np.pi/180
	return getattr(enlib.wcs, proj)(shape[-2:][::-1], box[:,::-1])

def spec2flat(shape, wcs, cov, exp=1.0):
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
	oshape= shape
	if len(oshape) == 2: oshape = (1,)+oshape
	ls  = np.sum(lmap(oshape, wcs)**2,0)**0.5
	# Translate from steradians to pixels
	cov = cov * np.prod(shape[-2:])/area(shape,wcs)
	if exp != 1.0: cov = multi_pow(cov, exp)
	cov[~np.isfinite(cov)] = 0
	cov   = cov[:oshape[-3],:oshape[-3]]
	return ndmap(enlib.utils.interpol(cov, np.reshape(ls,(1,)+ls.shape)),wcs)

def multi_pow(mat, exp, axes=[0,1]):
	"""Raise each sub-matrix of mat (ncomp,ncomp,...) to
	the given exponent in eigen-space."""
	res = enlib.utils.partial_expand(svd_pow(enlib.utils.partial_flatten(mat, axes, 0), exp), mat.shape, axes, 0)
	return samewcs(res, mat)

def downgrade(emap, factor):
	"""Returns enmap "emap" downgraded by the given integer factor
	(may be a list for each direction, or just a number) by averaging
	inside pixels."""
	factor = np.asarray(factor,dtype=int)
	if factor.ndim == 0: factor = np.array([factor,factor],dtype=int)
	tshape = emap.shape[-2:]/factor*factor
	res = np.mean(np.mean(np.reshape(emap[...,:tshape[0],:tshape[1]],emap.shape[:-2]+(tshape[0]/factor[0],factor[0],tshape[1]/factor[1],factor[1])),-1),-2)
	return ndmap(res, emap[...,::factor[0],::factor[1]].wcs)

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

def autocrop(m, method="plain", value=np.nan, margin=0, factors=None, return_info=False):
	"""Adjust the size of m to be more fft-friendly. If possible,
	blank areas at the edge of the map are cropped to bring us to a nice
	length. If there there aren't enough blank areas, the map is padded
	instead."""
	# Count blank areas on each side
	if np.isnan(value):
		hitmask = np.any(~np.isnan(m),axis=tuple(range(m.ndim-2)))
	else:
		hitmask = np.any(m!=value,axis=tuple(range(m.ndim-2)))
	hitrows = np.any(hitmask,1)
	hitcols = np.any(hitmask,0)
	blanks  = np.array([np.where(hitrows)[0][[0,-1]],np.where(hitcols)[0][[0,-1]]]).T
	blanks[1] = m.shape[-2:]-blanks[1]-1
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
	print "FIXME: grad not done"
	return np.reshape(np.real(ifft(fft(m)[None,:,...]*m.lmap()[:,None,...]*1j)),[2]+list(m.shape))

############
# File I/O #
############

def write_map(fname, emap, fmt=None):
	"""Writes an enmap to file. If fmt is not passed,
	the file type is inferred from the file extension, and can
	be either fits or hdf. This can be overriden by
	passing fmt with either 'fits' or 'hdf' as argument."""
	if fmt == None:
		if   fname.endswith(".hdf"):  fmt = "hdf"
		elif fname.endswith(".fits"): fmt = "fits"
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
		if   fname.endswith(".hdf"):  fmt = "hdf"
		elif fname.endswith(".fits"): fmt = "fits"
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
			wcs = enlib.wcs.from_bounds(data.shape[-2:][::-1], hfile["box"].value[:,::-1]*np.pi/180, csys)
			res = ndmap(data, wcs)
	if res.dtype.byteorder not in ['=','<' if sys.byteorder == 'little' else '>']:
		res = res.byteswap().newbyteorder()
	return res
