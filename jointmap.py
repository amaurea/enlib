import numpy as np, os, time, imp, copy, functools, sys
from scipy import ndimage, optimize, interpolate, integrate, stats, special
from enlib import enmap, retile, utils, bunch, cg, fft, powspec, array_ops, memory
from astropy import table
from astropy.io import fits
#from matplotlib import pyplot

def read_config(fname):
	config = imp.load_source("config", fname)
	config.path = fname
	return config

# OLD
def get_datasets(config, sel=None):
	# Set up boolean arrays for querys
	all_tags = set()
	for dataset in config.datasets:
		all_tags |= dataset.tags
	flags = {flag: np.array([flag in dataset.tags for dataset in config.datasets],bool) for flag in all_tags}
	# Extract the relevant datasets
	datasets = config.datasets
	if sel is not None:
		sel     = "&".join(["(" + w + ")" for w in utils.split_outside(sel, ",")])
		selinds = np.where(eval(sel, flags))[0]
		datasets = [datasets[i] for i in selinds]
	# Make all paths relative to us instead of the config file
	cdir = os.path.dirname(config.path)
	# In tiled input format, all input maps have the same geometry
	for dataset in datasets:
		for split in dataset.splits:
			split.map = os.path.join(cdir, split.map)
			split.div = os.path.join(cdir, split.div)
	# Read the geometry from all the datasets. Also make paths relative to us instead
	# of the config file
	for dataset in datasets:
		shape, wcs = read_geometry(dataset.splits[0].map)
		dataset.shape = shape
		dataset.wcs   = wcs
		dataset.beam  = read_beam(dataset.beam_params, workdir=cdir)
		dataset.box   = enmap.box(shape, wcs, corner=False)
	
		dataset.config= config # pretty backwards...
	return datasets

def read_geometry(fname):
	if os.path.isdir(fname):
		geo = retile.read_tileset_geometry(fname + "/tile%(y)03d_%(x)03d.fits")
		return geo.shape, geo.wcs
	else:
		return enmap.read_map_geometry(fname)

def read_beam(params, nl=50000, workdir="."):
	l = np.arange(nl).astype(float)
	if params[0] == "fwhm":
		sigma = params[1]*utils.fwhm*utils.arcmin
		return -0.5*l**2*sigma**2
	elif params[0] == "transfun":
		res   = np.zeros(nl)
		fname = os.path.join(workdir, params[1])
		bdata = np.loadtxt(fname)[:,1]
		ndata = len(bdata)
		res[:ndata] = np.log(bdata)
		# Fit power law to extend the beam beyond its end. That way we will have
		# a well-defined value everywhere.
		i1, i2 = ndata*18/20, ndata*19/20
		x1, x2 = np.log(l[i1]), np.log(l[i2])
		y1, y2 = res[i1], res[i2]
		alpha = (y2-y1)/(x2-x1)
		res[i2:] = res[i2] + alpha*np.log(l[i2:]/l[i2-1])
		return res
	else: raise ValueError("beam type '%s' not implemented" % type)

def beam_ratio(beam1, beam2): return beam1 - beam2

def beam_size(beam):
	"""Returns the l where the beam has fallen by one sigma. The beam should
	be given in the logarithmic form."""
	return np.where(beam > -1)[0][-1]

def eval_beam(beam, l, raw=False):
	res = utils.interpol(beam, l[None], order=1, mask_nan=False)
	if not raw: res = np.exp(res)
	return res

def calc_profile_ptsrc(freq, nl=50000):
	return np.full(nl, 1.0)

# This approach does not work as it is - it suffers from aliasing.
# It could probably be made to work with a fiducial beam, a variable
# step size, and more interpolation, but for now I have bypassed it
# to work directly in fourier space
#def calc_profile_sz(freq, scale=1.0, nl=50000, dl=250):
#	# This gives the y to signal response in fourier space
#	amp = sz_freq_core(freq*1e9)
#	# f(k) = fft(f(x)). Fourier interval is related to l
#	# via dl = 2*pi/(dx*n), and lmax is pi/dx. So dx must be
#	# pi/nl.
#	dl = dl/float(scale)
#	dr = np.pi/nl
#	nr = utils.nint(2*np.pi/(dr*dl))
#	r  = np.arange(nr)*dr # radians
#	# We want scale=1 to correspond to a FWHM of the sz profile at
#	# 1 arcmin. In dimensionless units the FWHM is 0.34208
#	y  = sz_rad_projected(r/utils.arcmin, scale)
#	fy = fft.rfft(y).real
#	l  = np.arange(len(fy),dtype=float)*dl
#	# Expand to full array
#	spline   = interpolate.splrep(l, np.log(fy))
#	profile  = np.exp(interpolate.splev(np.arange(nl), spline))
#	# This is a good approximation for avoiding aliasing. In the real world,
#	# the beam smoothes away high-l modes before they can be aliased, but since
#	# we don't have an easy way to compute the beam-convoluted real-space signal,
#	# we must avoid the aliasing another way
#	profile -= profile[-1]
#	# Normalize to 1 response in center, so the mean is not changed
#	profile /= profile[0]
#	return amp*profile

h_P = 6.62607004e-34
k_B =1.38064852e-23
def sz_freq_core(f, T=2.725):
	x  = h_P*f/(k_B*T)
	ex = np.exp(x)
	return x*(ex+1)/(ex-1)-4

def sz_rad_core(x):
	"""Dimensionless radial sz profile as a function of r/R500.
	Normalized to 1 in center."""
	c500  = 1.177
	alpha = 1.0510
	beta  = 5.4905
	gamma = 0.3081
	cx = c500*x
	p  = 1/(cx**gamma * (1+cx**alpha)**((beta-gamma)/alpha))
	return p

def sz_rad_projected_core(r_ort, zmax=1000):
	"""Dimensionaless angular sz profile with FWHM of 1 and an integral of 1. Takes about
	0.35 ms to evaluate on laptop."""
	return integrate.quad(lambda z: sz_rad_core((z**2+r_ort**2)**0.5*0.34208), 0, zmax)[0]*(2/3.6616198499715846)

class SZInterpolator:
	"""This class provides a fast spline-based interpolator for sz_rad_projected_core,
	which importantly makes it parallelizable. Only a single instance should be needed, which
	which is instantiated globally. When called it will set up its internal state the first
	time. The fwhm argument scales it radially while keeping the central amplitude constant.
	This means that the total integral will change. If this is not what you want, divide the
	result by fwhm**2. It takes 7 us to evaluate for 1 number, 77 us for an array of length
	1000 and 66 ms for an array of length 1e6.
	"""
	def __init__(self, rmax=1000, npoint=1001, sqrt=True, log=True):
		self.rmax, self.npoint, self.sqrt, self.log = rmax, npoint, sqrt, log
		self.spline = None
	def setup(self):
		if self.sqrt: self.r = np.linspace(0, self.rmax**0.5, self.npoint)**2
		else:         self.r = np.linspace(0, self.rmax, self.npoint)
		self.v = np.array([sz_rad_projected_core(r) for r in self.r])
		if self.log: self.spline = interpolate.splrep(self.r, np.log(self.v))
		else:        self.spline = interpolate.splrep(self.r, self.v)
	def __call__(self, r, fwhm=1.0, fwhm_deriv=False):
		"""If fwhm_deriv is True, then computes the derivative of the
		spline profile with respect to fwhm."""
		if self.spline is None: self.setup()
		x = r/fwhm
		if self.log:
			res = np.exp(interpolate.splev(x, self.spline))
			if fwhm_deriv:
				res *= -r/fwhm**2*interpolate.splev(x, self.spline, der=1)
		else:
			res = interpolate.splev(x, self.spline, der=fwhm_deriv)
			if fwhm_deriv:
				res *= -r/fwhm**2
		return res
sz_rad_projected_fast = SZInterpolator()

def sz_2d_profile(shape, pixshape, pos=[0,0], fwhm=1.0, oversample=5, core=10, pad=100,
		periodic=False, nofft=False, pos_deriv=None, scale_deriv=False):
	"""Return a 2d sz profile with shape [ny,nx] where each pixel has the physical
	shape pixshape [wy,wx] in arcminutes. This means that any cos(dec) factors must
	be included in pixshape. The profile will be offset by pos [oy,ox]
	from the fiducial location shape//2 in the center of the map. The sz profile
	will have the given fwhm in arcminutes. The profile will be evaluated at
	high resolution given by oversample in order to handle the sz profile's bandlimitlessness.
	This introduces a pixel window, which we deconvolve. Fractional offsets are supported
	and are handled with fourier shifting, which preserves the overall signal amplitude.
	With the default parameters, this function has an accuracy of 5e-6 relative to the peak
	for fwhm=1, and takes 14 ms to evaluate for a (33,33) map, about half of which is the
	fourier stuff.
	"""
	n   = np.array(shape[-2:])
	pos = np.array(pos)
	ipos = utils.nint(pos)
	fpos = pos-ipos
	bpos = n//2
	if periodic: pad = 0
	def over_helper(shape, pixshape, fwhm, offset, oversample):
		#print "over_hepler", shape, pixshape, fwhm, oversample
		oversample = oversample//2*2+1
		n = np.array(shape[-2:])
		N = n*oversample
		i = offset*oversample + oversample//2
		big_pos = (np.mgrid[:N[0],:N[1]] - i[:,None,None])*(np.array(pixshape)/oversample)[:,None,None]
		big_rad = np.sum(big_pos**2,0)**0.5
		big_map = sz_rad_projected_fast(big_rad, fwhm, fwhm_deriv=scale_deriv)
		# Then downgrade. This is done to get the pixel window. We can just use
		# fourier method to get the pixel window here, as the signal is not band-limited
		map   = enmap.downgrade(big_map, oversample)
		return map
	# First evaluate the locations at medium resolution
	map  = over_helper(shape, pixshape, fwhm, bpos+ipos, oversample)
	if nofft: return map
	# Then get the central region as high resolution
	while core >= 1:
		oversample *= 2
		cw   = core//2*2+1
		ci   = n//2-cw//2
		map[ci[0]:ci[0]+cw,ci[1]:ci[1]+cw] = over_helper((cw,cw), pixshape, fwhm, bpos+ipos-ci, oversample)
		core = core//2
	# We will now fourier shift to the target offset from the center, and also
	# deconvolve the band-limited part of the pixel window.
	pad_map = np.pad(map, pad, "constant")+0j
	fmap = fft.fft(pad_map, axes=[0,1])
	wy, wx = enmap.calc_window(fmap.shape)
	fmap *= wy[:,None]
	fmap *= wx[None,:]
	fmap  = fft.shift(fmap, fpos, axes=[0,1], nofft=True, deriv=pos_deriv)
	fft.ifft(fmap, pad_map, axes=[0,1], normalize=True)
	map[:] = pad_map[(slice(pad,-pad),)*pad_map.ndim].real if pad > 0 else pad_map.real
	return map

#def sz_rad_projected(r_ort, fwhm=1.0, xmax=5, dx=1e-3):
#	"""Compute the projected sz radial profile on the sky, as a function of
#	the angular distance r_ort from the center, in arcminutes. The cluster profile full-width
#	half-max is given by fwhm."""
#	x_ort = np.asarray(r_ort)*0.34208/fwhm
#	x_par = np.arange(0,xmax,dx)+dx/2
#	res   = np.array([np.sum(sz_rad_core((x_par**2 + xo**2)**0.5)) for xo in x_ort.reshape(-1)])
#	norm  = np.sum(sz_rad_core(x_par))
#	return res.reshape(x_ort.shape)/norm
#
#def sz_rad_projected_map(shape, wcs, fwhm=1.0, xmax=5, dx=1e-3):
#	pos = enmap.posmap(shape, wcs)
#	iy,ix = np.array(shape[-2:])//2+1
#	r   = np.sum((pos-pos[:,iy,ix][:,None,None])**2,0)**0.5
#	r   = np.roll(np.roll(r,-ix,-1),-iy,-2)
#	# Next build a 1d spline of the sz cluster profile covering all our radii
#	r1d = np.arange(0, np.max(r)*1.1, min(r[0,1],r[1,0])*0.5)
#	y1d = sz_rad_projected(r1d/utils.arcmin, fwhm, xmax, dx)
#	spline = interpolate.splrep(r1d, np.log(y1d))
#	y2d = enmap.ndmap(np.exp(interpolate.splev(r, spline)), wcs)
#	return y2d

def sz_map_profile(shape, wcs, fwhm=1.0):
	"""Evaluate the 2d sz profile for the enmap with the given shape
	and profile."""
	pixshape = enmap.pixshape(shape, wcs)/utils.arcmin
	pixshape[1] *= np.cos(enmap.pix2sky(shape, wcs, [shape[-2]/2,shape[-1]/2])[0])
	res = sz_2d_profile(shape, pixshape, fwhm=fwhm, oversample=1, core=50, periodic=True)
	# Shift center to top left corner
	res = np.roll(np.roll(res, -(shape[-2]//2), -2), -(shape[-1]//2), -1)
	return enmap.ndmap(res, wcs)

def butter(f, f0, alpha):
	if f0 <= 0: return f*0+1
	with utils.nowarn():
		return 1/(1 + (np.abs(f)/f0)**alpha)

def smooth_pix(map, pixrad):
	fmap  = enmap.fft(map)
	ky = np.fft.fftfreq(map.shape[-2])
	kx = np.fft.fftfreq(map.shape[-1])
	kr2   = ky[:,None]**2+kx[None,:]**2
	fmap *= np.exp(-0.5*kr2*pixrad**2)
	map   = enmap.ifft(fmap).real
	return map

def smooth_ps_pix_log(ps, pixrad):
	"""Smooth ps with 2 dof per pixel by the given pixel radius
	in lograthmic scale, while applying a correction factor to approximate
	the effect of plain smoothing when the background is flat."""
	return smooth_pix(ps, pixrad)*1.783

log_smooth_corrections = [ 1.0, # dummy for 0 dof
 3.559160, 1.780533, 1.445805, 1.310360, 1.237424, 1.192256, 1.161176, 1.139016,
 1.121901, 1.109064, 1.098257, 1.089441, 1.082163, 1.075951, 1.070413, 1.065836,
 1.061805, 1.058152, 1.055077, 1.052162, 1.049591, 1.047138, 1.045077, 1.043166,
 1.041382, 1.039643, 1.038231, 1.036866, 1.035605, 1.034236, 1.033090, 1.032054,
 1.031080, 1.030153, 1.029221, 1.028458, 1.027655, 1.026869, 1.026136, 1.025518,
 1.024864, 1.024259, 1.023663, 1.023195, 1.022640, 1.022130, 1.021648, 1.021144,
 1.020772]

def smooth_ps(ps, res, alpha=4, log=True, ndof=2):
	"""Smooth a 2d power spectrum to the target resolution in l"""
	# First get our pixel size in l
	lx, ly = enmap.laxes(ps.shape, ps.wcs)
	ires   = np.array([lx[1],ly[1]])
	smooth = np.abs(res/ires)
	# We now know how many pixels to somoth by in each direction,
	# so perform the actual smoothing
	if log: ps = np.log(ps)
	fmap  = enmap.fft(ps)
	ky    = np.fft.fftfreq(ps.shape[-2])
	kx    = np.fft.fftfreq(ps.shape[-1])
	fmap /= 1 + np.abs(2*ky[:,None]*smooth[0])**alpha
	fmap /= 1 + np.abs(2*kx[None,:]*smooth[1])**alpha
	ps    = enmap.ifft(fmap).real
	if log: ps = np.exp(ps)*log_smooth_corrections[ndof]
	return ps

def read_map(fname, pbox, name=None, cache_dir=None, dtype=None, read_cache=False):
	if os.path.isdir(fname):
		fname = fname + "/tile%(y)03d_%(x)03d.fits"
	if read_cache:
		map = enmap.read_map(cache_dir + "/" + name)
		if dtype is not None: map = map.astype(dtype)
	else:
		map = retile.read_area(fname, pbox).astype(dtype)
		if dtype is not None: map = map.astype(dtype)
		if cache_dir is not None and name is not None:
			enmap.write_map(cache_dir + "/" + name, map)
	#if map.ndim == 3: map = map[:1]
	return map

def map_fft(x): return enmap.fft(x)
def map_ifft(x): return enmap.ifft(x).real

def calc_pbox(shape, wcs, box, n=10):
	nphi = utils.nint(np.abs(360/wcs.wcs.cdelt[0]))
	dec = np.linspace(box[0,0],box[1,0],n)
	ra  = np.linspace(box[0,1],box[1,1],n)
	y   = enmap.sky2pix(shape, wcs, [dec,dec*0+box[0,1]])[0]
	x   = enmap.sky2pix(shape, wcs, [ra*0+box[0,0],ra])[1]
	x   = utils.unwind(x, nphi)
	pbox = np.array([
		[np.min(y),np.min(x)],
		[np.max(y),np.max(x)]])
	xm1 = np.mean(pbox[:,1])
	xm2 = utils.rewind(xm1, shape[-1]/2, nphi)
	pbox[:,1] += xm2-xm1
	pbox = utils.nint(pbox)
	return pbox

def make_dummy_tile(shape, wcs, box, pad=0, dtype=np.float64):
	pbox = calc_pbox(shape, wcs, box)
	if pad:
		pbox[0] -= pad
		pbox[1] += pad
	shape2, wcs2 = enmap.slice_geometry(shape, wcs, (slice(pbox[0,0],pbox[1,0]),slice(pbox[0,1],pbox[1,1])), nowrap=True)
	shape2 = shape[:-2]+tuple(pbox[1]-pbox[0])
	map = enmap.zeros(shape2, wcs2, dtype)
	div = enmap.zeros(shape2[-2:], wcs2, dtype)
	return bunch.Bunch(map=map, div=div)

def robust_ref(div,tol=1e-5):
	ref = np.median(div[div>0])
	ref = np.median(div[div>ref*tol])
	return ref

def add_missing_comps(map, ncomp, rms_factor=1e3):
	map  = map.preflat
	if len(map) >= ncomp: return map[:ncomp]
	omap = enmap.zeros((ncomp,)+map.shape[-2:], map.wcs, map.dtype)
	omap[:len(map)] = map[:ncomp]
	omap[len(map):] = np.random.standard_normal((ncomp-len(map),)+map.shape[-2:])*np.std(map)*rms_factor
	return omap

def common_geometry(geos, ncomp=None):
	shapes = np.array([shape[-2:] for shape,wcs in geos])
	assert np.all(shapes == shapes[0]), "Inconsistent map shapes"
	if ncomp is None:
		ncomps = np.array([shape[-3] for shape,wcs in geos if len(shape)>2])
		assert np.all(ncomps == ncomps[0]), "Inconsistent map ncomp"
		ncomp = ncomps[0]
	return (ncomp,)+tuple(shapes[0]), geos[0][1]

def filter_div(div):
	"""Downweight very thin stripes in the div - they tend to be problematic single detectors"""
	return enmap.samewcs(ndimage.minimum_filter(div, size=2), div)

dog = None
class JointMapset:
	def __init__(self, datasets, ffpad=0, ncomp=None):
		self.datasets = datasets
		self.ffpad    = ffpad
		self.shape, self.wcs = common_geometry([split.data.map.geometry for dataset in datasets for split in dataset.splits], ncomp=ncomp)
		self.dtype    = datasets[0].splits[0].data.map.dtype
		self.set_slice()
	@classmethod
	def read(cls, datasets, box, pad=0, verbose=False, cache_dir=None, dtype=None, div_unhit=1e-7, read_cache=False, ncomp=None, *args, **kwargs):
		odatasets = []
		for dataset in datasets:
			dataset = dataset.copy()
			pbox = calc_pbox(dataset.shape, dataset.wcs, box)
			#pbox = np.round(enmap.sky2pix(dataset.shape, dataset.wcs, box.T).T).astype(int)
			pbox[0] -= pad
			pbox[1] += pad
			psize = pbox[1]-pbox[0]
			ffpad = np.array([fft.fft_len(s, direction="above")-s for s in psize])
			pbox[1] += ffpad

			dataset.pbox = pbox
			osplits = []
			for split in dataset.splits:
				split = split.copy()
				if verbose: print "Reading %s" % split.map
				try:
					map = read_map(split.map, pbox, name=os.path.basename(split.map), cache_dir=cache_dir,dtype=dtype, read_cache=read_cache)
					div = read_map(split.div, pbox, name=os.path.basename(split.div), cache_dir=cache_dir,dtype=dtype, read_cache=read_cache).preflat[0]
				except IOError as e: continue
				map *= dataset.gain
				div *= dataset.gain**-2
				div[~np.isfinite(div)] = 0
				map[~np.isfinite(map)] = 0
				div[div<div_unhit] = 0
				if np.all(div==0): continue
				split.data = bunch.Bunch(map=map, div=div, empty=np.all(div==0))
				osplits.append(split)
			if len(osplits) < 2: continue
			dataset.splits = osplits
			odatasets.append(dataset)
		if len(odatasets) == 0: return None
		return cls(odatasets, ffpad, ncomp=ncomp, *args, **kwargs)

	def analyze(self, ref_beam=None, mode="weight", map_max=1e8, div_tol=20, apod_val=0.2, apod_alpha=5, apod_edge=120,
			beam_tol=1e-4, ps_spec_tol=0.5, ps_smoothing=20, ps_res=400, filter_kxrad=20, filter_highpass=200, filter_kx_ymax_scale=1, dewindow=False):
		# Find the typical noise levels. We will use this to decide where
		# divs and beams etc. can be truncated to improve convergence.
		datasets = self.datasets
		ncomp = max([split.data.map.preflat.shape[0] for dataset in datasets for split in dataset.splits])
		for dataset in datasets:
			for split in dataset.splits:
				split.ref_div = robust_ref(split.data.div)
				# Avoid single, crazy pixels
				split.data.div = np.minimum(split.data.div, split.ref_div*div_tol)
				split.data.div = filter_div(split.data.div)
				split.data.map = np.maximum(-map_max, np.minimum(map_max, split.data.map))
				# Expand map to ncomp components
				split.data.map = add_missing_comps(split.data.map, ncomp)
				if dewindow:
					split.data.map = enmap.apply_window(split.data.map, -1)
				# Build apodization
				apod = np.minimum(split.data.div/(split.ref_div*apod_val), 1.0)**apod_alpha
				apod.apod(apod_edge)
				split.data.div *= apod
				split.data.H   = split.data.div**0.5
			dataset.ref_div = np.sum([split.ref_div for split in dataset.splits])
		tot_ref_div = np.sum([dataset.ref_div for dataset in datasets])

		ly, lx   = enmap.laxes(self.shape, self.wcs)
		lr       = (ly[:,None]**2 + lx[None,:]**2)**0.5
		bmin = np.min([beam_size(dataset.beam) for dataset in datasets])
		# If no target beam was specified, skip putting them all on the same resolution
		if ref_beam is not None:
			# Deconvolve all the relative beams. These should ideally include pixel windows.
			# This could matter for planck
			for dataset in datasets:
				rel_beam  = beam_ratio(dataset.beam, ref_beam)
				# Avoid division by zero
				bspec     = np.maximum(eval_beam(rel_beam, lr), 1e-10)
				# We don't want to divide by tiny numbers, so we will cap the relative
				# beam. The goal is just to make sure that the deconvolved noise ends up
				# sufficiently high that anything beyond that is negligible. This will depend
				# on the div ratios between the different datasets. We can stop deconvolving
				# when beam*my_div << (tot_div-my_div). But deconvolving even by a factor
				# 1000 leads to strange numberical errors
				bspec = np.maximum(bspec, beam_tol*(tot_ref_div/dataset.ref_div-1))
				bspec_dec = np.maximum(bspec, 0.01)
				for split in dataset.splits:
					split.data.map = map_ifft(map_fft(split.data.map)/bspec_dec)
				# In theory we don't need to worry about the beam any more by this point.
				# But the pixel window might be unknown or missing. So we save the beam so
				# we can make sure the noise model makes sense
				dataset.bspec = bspec
				# We classify this as a low-resolution dataset if we did an appreciable amount of
				# deconvolution
				dataset.lowres = np.min(bspec) < 0.5
		else:
			dataset.bspec  = np.full(len(lr),1.0)
			dataset.lowres = False

		# Can now build the noise model and rhs for each dataset.
		# The noise model is N = HCH, where H = div**0.5 and C is the mean 2d noise spectrum
		# of the whitened map, after some smoothing.
		for dataset in datasets:
			nsplit = 0
			dset_map, dset_div = None, None
			for split in dataset.splits:
				if dset_map is None:
					dset_map = split.data.map*0
					dset_div = split.data.div*0
				dset_map += split.data.map * split.data.div
				dset_div += split.data.div
			# Form the mean map for this dataset
			dset_map[:,dset_div>0] /= dset_div[dset_div>0]
			# Then use it to build the diff maps and noise spectra
			dset_ps = None
			#i=0
			for split in dataset.splits:
				if split.data.empty: continue
				diff  = split.data.map - dset_map
				wdiff = diff * split.data.H
				#i+=1
				# What is the healthy area of wdiff? Wdiff should have variance
				# 1 or above. This tells us how to upweight the power spectrum
				# to take into account missing regions of the diff map.
				ndown = 10
				wvar  = enmap.downgrade(wdiff**2,ndown)
				goodfrac = np.sum(wvar > 1e-3)/float(wvar.size)
				if goodfrac < 0.1: goodfrac = 0
				ps    = np.abs(map_fft(wdiff))**2
				# correct for unhit areas, which can't be whitend
				with utils.nowarn(): ps   /= goodfrac
				if dset_ps is None:
					dset_ps = enmap.zeros(ps.shape, ps.wcs, ps.dtype)
				dset_ps += ps
				nsplit += 1
			if nsplit < 2: continue
			# With n splits, mean map has var 1/n, so diff has var (1-1/n) + (n-1)/n = 2*(n-1)/n
			# Hence tot-ps has var 2*(n-1)
			dset_ps /= 2*(nsplit-1)
			# Logarithmic smoothing is a sort of compromisbe between smoothing
			# N, which leaks the huge low-l noise peak into the surrounding area,
			# and smoothing iN, which fills in that noise hole. It works best if
			# the noise has an exponential profile, which it doesn't have - it's
			# a power law. But it's a 
			#dset_ps  = smooth_ps_pix_log(dset_ps, ps_smoothing)
			#dset_ps  = np.exp(smooth_pix(np.log(dset_ps), ps_smoothing))
			dset_ps  = smooth_ps(dset_ps, ps_res, ndof=2*(nsplit-1))
			if ref_beam is not None:
				# Use the beam we saved from earlier to make sure we don't have a remaining
				# pixel window giving our high-l parts too high weight. If everything has
				# been correctly deconvolved, we expect high-l dset_ps to go as
				# 1/beam**2. The lower ls will realistically be no lower than this either.
				# So we can simply take the max
				dset_ps_ref = np.min(np.maximum(dset_ps, dataset.bspec**-2*ps_spec_tol*0.1))
				dset_ps = np.maximum(dset_ps, dset_ps_ref*dataset.bspec**-2 * ps_spec_tol)
			# Our fourier-space inverse noise matrix is the inverse of this
			if np.all(np.isfinite(dset_ps)):
				iN = 1/dset_ps
			else:
				iN = enmap.zeros(dset_ps.shape, dset_ps.wcs, dset_ps.dtype)

			# Add any fourier-space masks to this
			if dataset.highpass:
				kxmask   = butter(lx, filter_kxrad,   -5)
				kxmask   = 1-(1-kxmask[None,:])*(np.abs(ly)<bmin*filter_kx_ymax_scale)[:,None]
				highpass = butter(lr, filter_highpass,-10)
				filter   = highpass * kxmask
				del kxmask, highpass
			else:
				filter   = 1
			if mode != "filter": iN *= filter
			dataset.iN     = iN
			dataset.filter = filter
			self.mode = mode

	def set_slice(self, slice=None):
		self.slice = slice
		if slice is None: slice = ""
		for dataset in self.datasets:
			# Clear existing selection
			for split in dataset.splits: split.active = False
			# Activate new selection
			inds = np.arange(len(dataset.splits))
			inds = eval("inds" + slice)
			for ind in inds:
				dataset.splits[ind].active = True

class AutoCoadder(JointMapset):
	def __init__(self, *args, **kwargs):
		JointMapset.__init__(self, *args, **kwargs)
	def calc_precon(self):
		datasets = self.datasets
		# Build the preconditioner
		self.tot_div = enmap.ndmap(np.sum([split.data.div for dataset in datasets for split in dataset.splits],0), self.wcs)
		self.tot_idiv = self.tot_div.copy()
		self.tot_idiv[self.tot_idiv>0] **=-1
		# Find the part of the sky hit by high-res data
		self.highres_mask = enmap.zeros(self.shape[-2:], self.wcs, np.bool)
		for dataset in datasets:
			if dataset.lowres: continue
			for split in dataset.splits:
				if split.data.empty or not split.active: continue
				self.highres_mask |= split.data.div > 0
	def calc_rhs(self):
		# Build the right-hand side. The right-hand side is sum(HNHm)
		rhs = enmap.zeros(self.shape, self.wcs, self.dtype)
		for dataset in self.datasets:
			#print "moo", dataset.name, "iN" in dataset, id(dataset)
			for split in dataset.splits:
				if split.data.empty or not split.active: continue
				w   = split.data.H*split.data.map
				fw  = map_fft(w)
				#print dataset.name
				fw *= dataset.iN
				if self.mode == "filter": fw *= dataset.filter
				w   = map_ifft(fw)*split.data.H
				rhs += w
		# Apply resolution mask
		rhs *= self.highres_mask
		self.rhs = rhs
	def A(self, x):
		m   = enmap.enmap(x.reshape(self.shape), self.wcs, copy=False)
		res = m*0
		for dataset in self.datasets:
			for split in dataset.splits:
				if split.data.empty or not split.active: continue
				w   = split.data.H*m
				w   = map_ifft(map_fft(w)*dataset.iN)
				w  *= split.data.H
				res += w
		# Apply resolution mask
		res *= self.highres_mask
		return res.reshape(-1)
	def M(self, x):
		m   = enmap.enmap(x.reshape(self.shape), self.wcs, copy=False)
		res = m * self.tot_idiv
		return res.reshape(-1)
	def solve(self, maxiter=100, cg_tol=1e-7, verbose=False, dump_dir=None):
		if np.sum(self.highres_mask) == 0: return None
		solver = cg.CG(self.A, self.rhs.reshape(-1), M=self.M)
		for i in range(maxiter):
			t1 = time.time()
			solver.step()
			t2 = time.time()
			if verbose:
				print "%5d %15.7e %5.2f" % (solver.i, solver.err, t2-t1)
			if dump_dir is not None and solver.i in [1,2,5,10,20,50] + range(100,10000,100):
				m = enmap.ndmap(solver.x.reshape(self.shape), self.wcs)
				enmap.write_map(dump_dir + "/step%04d.fits" % solver.i, m)
			if solver.err < cg_tol:
				if dump_dir is not None:
					m = enmap.ndmap(solver.x.reshape(self.shape), self.wcs)
					enmap.write_map(dump_dir + "/step_final.fits", m)
				break
		tot_map = self.highres_mask*solver.x.reshape(self.shape)
		tot_div = self.highres_mask*self.tot_div
		# Get rid of the fourier padding
		ny,nx = tot_map.shape[-2:]
		tot_map = tot_map[...,:ny-self.ffpad[0],:nx-self.ffpad[1]]
		tot_div = tot_div[...,:ny-self.ffpad[0],:nx-self.ffpad[1]]
		return bunch.Bunch(map=tot_map, div=tot_div)

class AutoFilter(JointMapset):
	def __init__(self, *args, **kwargs):
		JointMapset.__init__(self, *args, **kwargs)
	def analyze(self, inv_signal, cl_bg, dewindow=True, **kwargs):
		JointMapset.analyze(self, ref_beam=inv_signal, dewindow=dewindow, **kwargs)
		# The inverse signal profile in l-space
		self.inv_signal = np.array(inv_signal)
		# Deconvolve signal profile from background spectrum
		lmax_bg = min(len(cl_bg),len(inv_signal))
		self.cl_bg = cl_bg.copy()
		self.cl_bg[lmax_bg:] = 0
		self.cl_bg[:lmax_bg] *= inv_signal[:lmax_bg]**2
		# Evaluate 2d background spectrum
		self.S = enmap.spec2flat(self.shape, self.wcs, self.cl_bg[None,None])[0,0]
	def calc_precon(self):
		datasets = self.datasets
		# Build the preconditioner
		self.tot_div = enmap.ndmap(np.sum([split.data.div for dataset in datasets for split in dataset.splits],0), self.wcs)
		self.precon  = 1 # /(1+self.tot_div)
	def calc_rhs(self):
		# Build the right-hand side. The right-hand side is sum(HNHm)
		rhs = enmap.zeros(self.shape, self.wcs, self.dtype)
		for dataset in self.datasets:
			#print "moo", dataset.name, "iN" in dataset, id(dataset)
			for split in dataset.splits:
				if split.data.empty or not split.active: continue
				w   = split.data.H*split.data.map
				fw  = map_fft(w)
				fw *= dataset.iN
				if self.mode == "filter": fw *= dataset.filter
				w   = map_ifft(fw)*split.data.H
				rhs += w
		self.rhs = rhs
	def A(self, x):
		m   = enmap.enmap(x.reshape(self.shape), self.wcs, copy=False)
		Sm  = map_ifft(self.S*map_fft(m))
		res = m*0
		for dataset in self.datasets:
			for split in dataset.splits:
				if split.data.empty or not split.active: continue
				w   = split.data.H*Sm
				w   = map_ifft(map_fft(w)*dataset.iN)
				w  *= split.data.H
				res += w
		res += m
		return res.reshape(-1)
	def M(self, x):
		return x.copy()
		#m   = enmap.enmap(x.reshape(self.shape), self.wcs, copy=False)
		#res = m * self.precon
		#return res.reshape(-1)
	def solve(self, maxiter=100, cg_tol=1e-7, verbose=False, dump_dir=None):
		solver = cg.CG(self.A, self.rhs.reshape(-1), M=self.M)
		for i in range(maxiter):
			t1 = time.time()
			solver.step()
			t2 = time.time()
			if verbose:
				print "%5d %15.7e %5.2f" % (solver.i, solver.err, t2-t1)
			if dump_dir is not None and solver.i in [1,2,5,10,20,50] + range(100,10000,100):
				m = enmap.ndmap(solver.x.reshape(self.shape), self.wcs)
				print calc_div_noise_normalization(m, self.tot_div)
				enmap.write_map(dump_dir + "/step%04d.fits" % solver.i, m)
			if solver.err < cg_tol:
				if dump_dir is not None:
					m = enmap.ndmap(solver.x.reshape(self.shape), self.wcs)
					enmap.write_map(dump_dir + "/step_final.fits", m)
				break
		tot_map = solver.x.reshape(self.shape)
		# Get rid of the fourier padding
		ny,nx = tot_map.shape[-2:]
		tot_map = tot_map[...,:ny-self.ffpad[0],:nx-self.ffpad[1]]
		return bunch.Bunch(map=tot_map)

def calc_div_noise_normalization(map, div, bsize=45, nsigma=4):
	"""Fit a model var = alpha*div + beta to the first component of map"""
	nby = map.shape[-2]//bsize
	nbx = map.shape[-1]//bsize
	def blockify(m):
		res = np.array(m.preflat[0,:nby*bsize,:nbx*bsize].reshape(nby,bsize,nbx,bsize))
		return np.transpose(res, (0, 2, 1, 3)).reshape(nby*nbx,-1)
	map = blockify(map)
	div = blockify(div)
	# Cap outliers in each bock
	for b in map:
		ulim= np.percentile(b,75)*3
		llim= np.percentile(b,25)*3
		b[b>ulim] = ulim
		b[b<llim] = llim
	y   = np.mean(map**2,1)
	print "y"
	print np.median(y), np.mean(y)
	x   = np.mean(div,1)
	def calc_model(coeff): return np.exp(coeff[0])*x/(1+x*np.exp(coeff[1]))**2
	def calc_chisq(coeff): return np.sum((y-calc_model(coeff))**2)
	coeff0 = np.log(np.array([1e-3,100**2]))
	coeffs = optimize.fmin_powell(calc_chisq, coeff0, disp=False)
	model  = calc_model(coeffs)
	enmap.write_map("y.fits", y.reshape(nby,nbx))
	enmap.write_map("model.fits", model.reshape(nby,nbx))
	enmap.write_map("ratio.fits", (y/model).reshape(nby,nbx))
	enmap.write_map("diff.fits", (y-model).reshape(nby,nbx))
	return np.exp(coeffs)


# va cov is (1+N"S)"N"(1+N"S)'. Assume pixel diag approximation of S is S| and pixel
# diag approximation of N" is div. Then this cov's diag is approximately
# div/(1+div*S|)**2. When div is big, this goes as 1/div, but when it is small it
# goes as div. Requires nonlinear fit

#################################
##### New stuff below here ######
#################################

# The inheritance stuff above is much less useful than I thought it would be.
# I feel like I end up fighting against it. I'll build my new stuff without it,
# and hopefully transition everything away from it eventually.

class Mapset:
	def __init__(self, config, sel=None):
		self.config = config
		self.select(sel)
	def select(self, sel):
		config   = self.config
		all_tags = set()
		for dataset in config.datasets:
			all_tags |= dataset.tags
		flags = {flag: np.array([flag in dataset.tags for dataset in config.datasets],bool) for flag in all_tags}
		# Extract the relevant datasets
		datasets = config.datasets
		if sel is not None:
			sel     = "&".join(["(" + w + ")" for w in utils.split_outside(sel, ",")])
			selinds = np.where(eval(sel, flags))[0]
			datasets = [datasets[i] for i in selinds]
		# Make all paths relative to us instead of the config file
		cdir = os.path.dirname(config.path)
		# In tiled input format, all input maps have the same geometry
		for dataset in datasets:
			for split in dataset.splits:
				split.map = os.path.join(cdir, split.map)
				split.div = os.path.join(cdir, split.div)
		# Read the geometry from all the datasets. Also make paths relative to us instead
		# of the config file
		self.nl = 30000
		for dataset in datasets:
			shape, wcs = read_geometry(dataset.splits[0].map)
			dataset.shape = shape
			dataset.wcs   = wcs
			dataset.beam  = read_beam(dataset.beam_params, workdir=cdir, nl=self.nl)
			dataset.box   = enmap.box(shape, wcs, corner=False)
		self.datasets = datasets
	def copy(self):
		config = self.config
		self.config = None
		res = copy.deepcopy(self)
		res.config = config
		self.config = config
		return res
	def read(self, box, pad=0, prune=True, verbose=False, cache_dir=None, dtype=np.float64, div_unhit=1e-7, read_cache=False, ncomp=1, wcs=None):
		"""Read the data for each of our datasets that falls within the given box, returning a new Mapset
		with the mapset.datasets[:].split[:].data member filled with a map and div. If prune is False,
		then the returned mapset will have the same maps as the original mapset. If prune is True (the default),
		on the other hand, splits with no data are removed, as are datasets with too few splits. This can
		result in all the data being removed, in which case None is returned."""
		res = self.copy()
		res.ffpad, res.shape, res.wcs = None, None, None
		res.ncomp, res.dtype = ncomp, dtype
		for dataset in res.datasets:
			# Find the pixel coordinates of our tile
			pbox = calc_pbox(dataset.shape, dataset.wcs, box)
			pbox[0] -= pad
			pbox[1] += pad
			# Determine the optimal fourier padding
			psize = pbox[1]-pbox[0]
			ffpad = np.array([fft.fft_len(s, direction="above")-s for s in psize])
			pbox[1] += ffpad
			# The overall tile geometry (wcs in particular) can be found even if there isn't
			# any data to read, but in this case we can't be sure that the wcs object we read
			# has the reference pixel placed sensibly. That's why we cal fix_wcs here.
			if res.shape is None:
				res.shape, res.wcs = enmap.slice_geometry(dataset.shape, dataset.wcs, (slice(pbox[0,0],pbox[1,0]),slice(pbox[0,1],pbox[1,1])), nowrap=True)
				res.wcs   = enmap.enlib.wcs.fix_wcs(res.wcs)
				res.shape = (ncomp,)+res.shape[-2:]
				res.ffpad = ffpad
			dataset.pbox  = pbox
			dataset.ngood = 0

			# Reading lots of uncessessary maps is slow. Should otpimize read_map.
			# But as long as we are allowed to completely skip datasets (prune=True),
			# we can just skip datasets that we know are empty.
			if pbox_out_of_bounds(pbox, dataset.shape, dataset.wcs) and prune:
				continue

			for si, split in enumerate(dataset.splits):
				split.data = None
				if verbose: print "Reading %s" % split.map
				try:
					map = read_map(split.map, pbox, name=os.path.basename(split.map), cache_dir=cache_dir,dtype=dtype, read_cache=read_cache)
					div = read_map(split.div, pbox, name=os.path.basename(split.div), cache_dir=cache_dir,dtype=dtype, read_cache=read_cache).preflat[0]
				except IOError as e: continue
				map *= dataset.gain
				div *= dataset.gain**-2
				div[~np.isfinite(div)] = 0
				map[~np.isfinite(map)] = 0
				div[div<div_unhit] = 0
				if np.all(div==0): continue
				split.data = bunch.Bunch(map=map, div=div)
				dataset.ngood += 1
		if prune:
			res.datasets = [dataset for dataset in res.datasets if dataset.ngood >= 2]
			for dataset in res.datasets:
				dataset.splits = [split for split in dataset.splits if split.data is not None]
		# Precompute our lmap
		res.l = enmap.modlmap(res.shape, res.wcs)
		# At this point we have read all the data, but it's possible that we didn't actually
		# read anything useful. If so res.datasets can be empty, or invididual datasets' ngood may be 0
		return res

def sanitize_maps(mapset, map_max=1e8, div_tol=20, apod_val=0.2, apod_alpha=5, apod_edge=60, apod_div_edge=10, crop_div_edge=0):
	"""Get rid of extreme values in maps and divs, and further downweights the the edges and
	faint regions of div."""
	for dataset in mapset.datasets:
		for i, split in enumerate(dataset.splits):
			if split.data is None: continue
			split.ref_div = robust_ref(split.data.div)
			# Avoid single, crazy pixels
			split.data.div = np.minimum(split.data.div, split.ref_div*div_tol)
			split.data.div = filter_div(split.data.div)
			split.data.map = np.maximum(-map_max, np.minimum(map_max, split.data.map))
			if crop_div_edge:
				# Avoid areas too close to the edge of div
				split.data.div *= ndimage.distance_transform_edt(split.data.div > 0) > crop_div_edge
			# Expand map to ncomp components
			split.data.map = add_missing_comps(split.data.map, mapset.ncomp)
			# Distrust very low hitcount regions
			split.data.apod  = np.minimum(split.data.div/(split.ref_div*apod_val), 1.0)**apod_alpha
			# Distrust regions very close to the edge of the hit area
			split.data.apod *= apod_mask_edge(split.data.div > split.ref_div*1e-2, apod_div_edge)
			# Make things more fourier-friendly
			split.data.apod *= split.data.apod.apod(apod_edge)
			# And apply it
			split.data.div *= split.data.apod
		dataset.ref_div = np.sum([split.ref_div for split in dataset.splits if split.data is not None])
	mapset.ref_div = np.sum([dataset.ref_div for dataset in mapset.datasets])
	mapset.apod_edge = apod_edge

def build_noise_model(mapset, ps_res=400, filter_kxrad=20, filter_highpass=200, filter_kx_ymax_scale=1):
	"""This assumes that the mapset has been pruned, and may further prune the result"""
	for dataset in mapset.datasets:
		nsplit = 0
		dset_map, dset_div = None, None
		dataset.iN = None
		for split in dataset.splits:
			if split.data is None: continue
			if dset_map is None:
				dset_map = split.data.map*0
				dset_div = split.data.div*0
			dset_map += split.data.map * split.data.div
			dset_div += split.data.div
			# Also form the pixel part of our noise model
			split.data.H  = split.data.div**0.5
		# Form the mean map for this dataset
		dset_map[:,dset_div>0] /= dset_div[dset_div>0]
		#enmap.write_map("test_totmap.fits", dset_map)
		# Then use it to build the diff maps and noise spectra
		dset_ps = None
		for i, split in enumerate(dataset.splits):
			if split.data is None: continue
			#enmap.write_map("test_map_%s_%d.fits" % (dataset.name, i), split.data.map)
			#enmap.write_map("test_div_%s_%d.fits" % (dataset.name, i), split.data.div)
			diff  = split.data.map - dset_map
			#enmap.write_map("test_diff_%s_%d.fits" % (dataset.name, i), diff)
			# We can't whiten diff with just H.
			# diff = m_i - sum(div)"sum(div_j m_j), and so has
			# var  = div_i" + sum(div)"**2 * sum(div) - 2*sum(div)"div_i/div_i
			#      = div_i" - sum(div)"
			# ivar = (div_i" - sum(div)")"
			with utils.nowarn():
				diff_H   = (1/split.data.div - 1/dset_div)**-0.5
			diff_H[~np.isfinite(diff_H)] = 0
			wdiff = diff * diff_H
			#enmap.write_map("test_wdiff_%s_%d.fits" % (dataset.name, i), wdiff)
			# What is the healthy area of wdiff? Wdiff should have variance
			# 1 or above. This tells us how to upweight the power spectrum
			# to take into account missing regions of the diff map.
			ndown = 10
			wvar  = enmap.downgrade(wdiff**2,ndown)
			goodfrac_var  = np.sum(wvar > 1e-3)/float(wvar.size)
			goodfrac_apod = np.mean(split.data.apod)
			goodfrac = min(goodfrac_var, goodfrac_apod)
			if goodfrac < 0.1: continue
			ps    = np.abs(map_fft(wdiff))**2
			#enmap.write_map("test_ps_raw_%s_%d.fits" % (dataset.name, i), ps)
			# correct for unhit areas, which can't be whitend
			with utils.nowarn(): ps   /= goodfrac
			if dset_ps is None:
				dset_ps = enmap.zeros(ps.shape, ps.wcs, ps.dtype)
			dset_ps += ps
			nsplit += 1
		if nsplit < 2: continue
		dset_ps /= nsplit
		#enmap.write_map("test_ps_raw_%s.fits" % dataset.name, dset_ps)
		# Smooth ps to reduce sample variance
		dset_ps  = smooth_ps(dset_ps, ps_res, ndof=2*(nsplit-1))
		print "dset_ps", np.sum(dset_ps), dataset.name
		# Apply noise window correction if necessary:
		noisewin = dataset.noise_window_params[0] if "noise_window_params" in dataset else "none"
		if   noisewin == "none": pass
		elif noisewin == "lmax":
			# ps beyond lmax is not valid. Use values near lmax to extrapolate
			lmax = dataset.noise_window_params[1]
			lref = lmax*3/4
			refval = np.mean(dset_ps[:,(mapset.l>=lref)&(mapset.l<lmax)],1)
			dset_ps[:,mapset.l>=lmax] = refval[:,None]
		elif noisewin == "separable":
			# The map has been interpolated using something like bicubic interpolation,
			# leading to an unknown but separable pixel window
			ywin, xwin = estimate_separable_pixwin_from_normalized_ps(dset_ps[0])
			print "ywin", utils.minmax(ywin), "xwin", utils.minmax(xwin), dataset.name
			ref_area = (ywin[:,None] > 0.9)&(xwin[None,:] > 0.9)&(dset_ps[0]<2)
			print "ref_ara", np.sum(ref_area), dataset.name
			if np.sum(ref_area) == 0: ref_area[:] = 1
			dset_ps /= ywin[:,None]**2
			dset_ps /= xwin[None,:]**2
			dset_ps[:,(ywin[:,None]<0.25)|(xwin[None,:]<0.25)] = np.mean(dset_ps[:,ref_area],1)
			# Store the separable window so it can be used for the beam too
			dataset.ywin, dataset.xwin = ywin, xwin
		else: raise ValueError("Noise window type '%s' not supported" % noisewin)
		#enmap.write_map("test_ps_smooth_%s.fits" % dataset.name, dset_ps)
		# If we have invalid values, then this whole dataset should be skipped
		print "finite?", np.all(np.isfinite(dset_ps)), dataset.name
		if not np.all(np.isfinite(dset_ps)): continue
		dataset.iN  = 1/dset_ps
	# Prune away bad datasets and splits
	for dataset in mapset.datasets:
		dataset.splits = [split for split in dataset.splits if split.data is not None]
	mapset.datasets = [dataset for dataset in mapset.datasets if len(dataset.splits) >= 2 and dataset.iN is not None]

def setup_filter(mapset, mode="weight", filter_kxrad=20, filter_highpass=200, filter_kx_ymax_scale=1):
	# Add any fourier-space masks to this
	ly, lx  = enmap.laxes(mapset.shape, mapset.wcs)
	lr      = (ly[:,None]**2 + lx[None,:]**2)**0.5
	if len(mapset.datasets) > 0:
		bmin = np.min([beam_size(dataset.beam) for dataset in mapset.datasets])
	for dataset in mapset.datasets:
		if dataset.highpass:
			kxmask   = butter(lx, filter_kxrad,   -5)
			kxmask   = 1-(1-kxmask[None,:])*(np.abs(ly)<bmin*filter_kx_ymax_scale)[:,None]
			highpass = butter(lr, filter_highpass,-10)
			filter   = highpass * kxmask
			del kxmask, highpass
		else:
			filter   = 1
		if mode != "filter": dataset.iN *= filter
		dataset.filter = filter
	mapset.mode = mode

def downweight_lowl(mapset, lknee, alpha, lim=1e-10):
	"""Inflate low-l noise below l=lknee uniformly for all datasets.
	This could be used to represent a generic low-l foreground component
	with unknown frequency dependence, for example."""
	with utils.nowarn():
		filter = 1/(1+(mapset.l/lknee)**-alpha)
	filter = np.maximum(filter, lim)
	for dataset in mapset.datasets:
		dataset.iN *= filter

def setup_profiles_ptsrc(mapset):
	setup_profiles_helper(mapset, lambda freq: calc_profile_ptsrc(freq, nl=mapset.nl))
def setup_profiles_sz(mapset, scale):
	# sz must be evaluated in real space to avoid aliasing. First get the
	# distance from the corner, including wrapping. We do this by getting the
	# distance from the center, and then moving the center to the corner afterwards
	y2d = sz_map_profile(mapset.shape, mapset.wcs, fwhm=scale)
	y2d_harm  = np.abs(map_fft(y2d))
	y2d_harm /= y2d_harm[0,0]
	# We can now scale it as appropriately for each dataset
	cache = {}
	for d in mapset.datasets:
		if d.freq not in cache:
			cache[d.freq] = y2d_harm * sz_freq_core(d.freq*1e9)
		d.signal_profile_2d = cache[d.freq]

def setup_profiles_helper(mapset, fun):
	cache = {}
	for d in mapset.datasets:
		if d.freq not in cache:
			profile_1d = fun(d.freq)
			profile_2d = ndimage.map_coordinates(profile_1d, mapset.l[None], order=1)
			cache[d.freq] = [profile_1d,profile_2d]
		d.signal_profile, d.signal_profile_2d = cache[d.freq]

def setup_beams(mapset):
	"""Set up the full beams with pixel windows for each dataset in the mapset"""
	cache = {}
	for d in mapset.datasets:
		param = (d.beam_params, d.pixel_window_params)
		if param not in cache:
			beam_2d = eval_beam(d.beam, mapset.l)
			# Apply pixel window
			if d.pixel_window_params[0] == "native":
				wy, wx = enmap.calc_window(beam_2d.shape)
				beam_2d *= wy[:,None]
				beam_2d *= wx[None,:]
			elif d.pixel_window_params[0] == "none":
				pass
			elif d.pixel_window_params[0] == "lmax":
				beam_2d[mapset.l>d.pixel_window_params[1]] = 0
			elif d.pixel_window_params[0] == "separable":
				try:
					beam_2d *= d.ywin[:,None]
					beam_2d *= d.xwin[None,:]
				except AttributeError:
					print "Automatic separable pixel window can only be used together with"
					print "the corresponding automatic spearable noise pixel window"
					raise
			else: raise ValueError("Unrecognized pixel window type '%s'" % (d.pixel_window_params[0]))
			cache[param] = beam_2d
		d.beam_2d = cache[param]

def setup_background_cmb(mapset, cl):
	# The background is shared between all datasets
	cl_TT = cl.reshape(-1,cl.shape[-1])[0]
	mapset.S = enmap.spec2flat(mapset.shape, mapset.wcs, cl_TT[None,None])[0,0]

def setup_target_beam(mapset, beam=None):
	"""Set up the target beam for map coadding. If no beam (which should be 2d)
	is passed in, then it is computed as the best of all input beams"""
	if beam is None:
		beam = mapset.datasets[0].beam_2d.copy()
		for dataset in mapset.datasets[1:]:
			beam = np.maximum(beam, dataset.beam_2d)
	mapset.target_beam = beam

def setup_mask_common_lowres(mapset, mask):
	"""Set up our mask if any. For now, we assume a common mask for."""
	if mask is None: return
	mask_patch = enmap.project(mask, mapset.shape, mapset.wcs, order=1, mask_nan=False)
	for dataset in mapset.datasets:
		dataset.mask = mask_patch
		for split in dataset.splits:
			split.data.div *= 1-mask_patch

class SignalFilter:
	def __init__(self, mapset):
		self.mapset = mapset
		# Extract and flatten all our input maps.
		self.m  = [split.data.map.preflat[0]  for dataset in mapset.datasets for split in dataset.splits]
		self.H  = [split.data.H               for dataset in mapset.datasets for split in dataset.splits]
		self.iN = [dataset.iN.preflat[0]      for dataset in mapset.datasets for split in dataset.splits]
		self.hN = [dataset.iN.preflat[0]**0.5 for dataset in mapset.datasets for split in dataset.splits]
		self.B  = [dataset.beam_2d            for dataset in mapset.datasets for split in dataset.splits]
		self.shape, self.wcs = mapset.shape[-2:], mapset.wcs
		self.dtype= mapset.dtype
		self.ctype= np.result_type(self.dtype,0j)
		self.npix = self.shape[0]*self.shape[1]
		self.nmap = len(self.m)
		self.tot_div = enmap.zeros(self.shape, self.wcs, self.dtype)
		for H in self.H: self.tot_div += H**2
	# Q is only needed for alpha, so defer initialization of it to make it easy to
	# compute alpha for many different profiles for the same mus.
	@property
	def Q(self): return [d.signal_profile_2d  for d in self.mapset.datasets for s in d.splits]
	def calc_rhs(self):
		# Calc rhs = Ch H m. This is effectively a set of fully whitened maps.
		# rhs is returend in fourier space
		rhs = [self.hN[i]*map_fft(self.H[i]*self.m[i]) for i in range(self.nmap)]
		return rhs
	def calc_mu(self, rhs, maxiter=250, cg_tol=1e-4, verbose=False, dump_dir=None):
		if self.nmap == 0: return enmap.zeros((self.nmap,)+self.shape, self.wcs, self.dtype)
		# Note: In the stuff below, the B in P is block-diagonal while the B before s is broadcasting.
		# Should have used a different notation for this.
		# m = Pa + Bs + n = Pa + ntot, Ntot = BSB' + N
		# a = (P'Ntot"P)"P'Ntot"m <=>
		# P'Ntot"P a = P'Ntot"m, where A = cov(a) = (P'Ntot"P)"
		# But we don't want a, we want ia = A"a = P'Ntot"m = P' im, with Ntot im = (BSB'+N) im = m
		# N" = HCH => N = H"C"H" (block diagonal, invertible)
		# (BSB'+H"C"H") im = m
		# Ch H (BSB'+H"C"H") HCh (Ch"H") im = Ch H m
		# (Ch H BSB' H Ch + 1) mu = rhs, with mu = Ch"H" im <=> im = H Ch mu; and rhs = Ch H m
		# mu is returned in fourier space
		def zip(maps): return np.concatenate([m.reshape(-1) for m in maps]).view(self.dtype)
		def unzip(x): return enmap.ndmap(x.view(self.ctype).reshape((-1,)+self.shape),self.wcs)
		def A(x):
			fmaps  = unzip(x)
			ofmaps = [f.copy() for f in fmaps]
			# Compute SB'HCh
			SBHCh = enmap.zeros(self.shape, self.wcs, self.ctype)
			for i in range(self.nmap):
				SBHCh += self.B[i]*map_fft(self.H[i]*map_ifft(self.hN[i]*fmaps[i]))
			SBHCh *= self.mapset.S
			# Complete ChHBSB'HCh and add it to omaps
			for i in range(self.nmap):
				ofmaps[i] += self.hN[i]*map_fft(self.H[i]*map_ifft(self.B[i]*SBHCh))
			return zip(ofmaps)
		# How can we build an efficient preconditioner for CMB dominated cases?
		# Assume harmonic only first. Then each fourier bin becomes a small nmap x nmap equation
		# system. But nmap x nmap can actually be pretty big. Can we do better?
		# We want (1+VV')" = 1 - V(1+V'V)"V' using woodbury. This is not that hard to
		# compute
		Hmean  = [np.mean(H) for H in self.H]
		V = enmap.zeros((self.nmap,)+self.shape, self.wcs, self.dtype)
		for i in range(self.nmap):
			V[i] = self.mapset.S**0.5*self.B[i]*Hmean[i]*self.hN[i]
		prec_core = 1/(1+np.sum(V**2,0))
		def M(x):
			fmaps = unzip(x)
			omaps = fmaps - V*prec_core*np.sum(V*fmaps,0)
			return zip(omaps)
		solver = cg.CG(A, zip(rhs), M=M)
		for i in range(maxiter):
			t1 = time.time()
			solver.step()
			t2 = time.time()
			if verbose:
				print "%5d %15.7e %5.2f" % (solver.i, solver.err, t2-t1)
			if dump_dir is not None and solver.i in [1,2,5,10,20,50] + range(100,10000,100):
				for j,m in enumerate(unzip(solver.x)):
					enmap.write_map(dump_dir + "/step%04d_mu%04d.fits" % (solver.i,j), map_ifft(m))
			if solver.err < cg_tol: break
		tot_mu = unzip(solver.x)
		if dump_dir is not None:
			for j,m in enumerate(tot_mu):
				enmap.write_map(dump_dir + "/step_final_mu%04d.fits" % j, map_ifft(m))
		return tot_mu
	def calc_alpha(self, mu):
		# Compute alpha = P'H Ch mu = A"a = P'N"m. The result will be in real space.
		alpha = enmap.zeros(self.shape, self.wcs, self.ctype)
		for i in range(self.nmap):
			alpha += self.Q[i]*self.B[i]*map_fft(self.H[i]*map_ifft(self.hN[i]*mu[i]))
		alpha = map_ifft(alpha)
		return alpha
	def calc_alpha_simple(self):
		# Plain matched filter for a single map
		Hmean = np.median(self.H[0])
		Ntot  = self.mapset.S*self.B[0]**2 + 1/np.maximum(Hmean**2*self.iN[0],1e-25)
		filter= self.Q[0]*self.B[0]/Ntot
		return map_ifft(filter*map_fft(self.m[0])), Ntot
	def calc_dalpha_empirical(self, alpha, mask_pad=10, downgrade=4):
		"""Fit the variance of alpha as a linear combination of individual divs,
		and return the best-fit rms map."""
		if len(self.H) == 0: return alpha*0+np.inf
		B     = np.array(self.H)**2
		# Mask out underexposed regions
		ref_val= np.percentile(self.tot_div,95)
		apod   = (alpha*0+1).apod(self.mapset.apod_edge)
		mask   = apod > 0.8
		# I don't trust the edge
		mask  &= ndimage.distance_transform_edt(self.tot_div > ref_val*0.01) > mask_pad
		mask  &= np.any(B>0,0)
		dmask  = enmap.downgrade(mask, downgrade) == 1
		#Bs     = B[:,mask]
		#d      = alpha[mask]**2
		with utils.nowarn():
			Bs     = 1/enmap.downgrade(1/B, downgrade)[:,dmask]
		d      = enmap.downgrade(alpha**2, downgrade)[dmask]
		def calc_chisq2(a):
			resid  = np.log(d)-np.log(a.dot(Bs))
			chisq  = np.sum(resid**2)
			dres   = -2*Bs/a.dot(Bs)
			dchisq = dres.dot(resid)
			return chisq, dchisq
		# Initial value is the independent guess
		a0 = np.maximum(np.sum(Bs*d,1)/np.sum(Bs**2,1),0)
		a, chisq, info  = optimize.fmin_l_bfgs_b(calc_chisq2, a0, bounds=[[1e-8,np.inf]]*len(a0))
		model      = np.einsum("i,ijk->jk", a, B)
		alpha_rms  = enmap.samewcs(model**0.5, alpha)
		norm       = utils.medmean((alpha[mask]/alpha_rms[mask])**2,1e-2)**0.5
		alpha_rms *= norm
		#enmap.write_map("alpha_rms.fits", alpha_rms)
		# Downweight the apodized region at the edge, and completley distrust the masked values
		with utils.nowarn():
			alpha_rms /= apod
		alpha_rms[~mask] = np.inf
		return alpha_rms
	def calc_filter_harmonic(self):
		# The harmonic-only representation of the total filter
		# F such that alpha = Fm, e.g. F = P'Ntot" = P'(BSB'+H"C"H")"
		# sim P'HCH - P'HCHBSh(1+ShB'HCHBSh)"ShB'HCH
		# Since there are multiple ms the result is one filter profile for
		# each. But these can be averaged to get a single overall profile.
		#
		# What do I get for 2 freq, harm-only case?
		#
		# [P1' P2'] [ S+N1  S  ]" [m1] = 1/((S+N1)*(S+N2)-S**2) * [P1' P2'] [S+N2 -S  ] [m1]
		#           [ S   S+N2 ]  [m2]                                      [ -S  S+N1] [m2]
		#
		# = (N1 N2 + S*(N1+N2))" [P1' P2'] [(S+N2)*m1 - S*m2]
		#                                  [(S+N1)*m2 - S*m1]
		#
		# = (N1*N2 + S*(N1+N2))"*(P1*[(S+N2)*m1-S*m2] + P2*[(S+N1)*m2-S*m1])
		# = -- * [(P1-P2)*S*(m1-m2) + P1*N2*m1 + P2*N1*m2]
		# = (1+S*(N1"+N2"))" * [(P1-P2)*S*N1"*N2"*(m1-m2) + P1*N1"*m1 + P2*N2"*m2]
		# Case 1: S-dominated
		#  (N1+N2)"(P1-P2)*(m1-m2)
		# Case 2: N-dominated
		#  P1*N1"*m1 + P2*N2"*m2
		# Case 3: Intermediate
		#  [P1*N1"*m1+P2*N2"*m2]/(S*(N1"+N2"))
		# The only term that benefits from multifreq is Case 1.

		# P'C(1-SC(1 + S sum(C))")
		Hmean = [np.mean(H) for H in self.H]
		sumPHCH = enmap.zeros(self.shape, self.wcs, self.dtype)
		PHCHB   = enmap.zeros(self.shape, self.wcs, self.dtype)
		sumHCHB = enmap.zeros(self.shape, self.wcs, self.dtype)
		core  = enmap.zeros(self.shape, self.wcs, self.dtype)
		for i in range(self.nmap):
			PHCHB   += self.Q[i]*self.B[i]**2*Hmean[i]**2*self.iN[i]
			# These two represent adding together the filters that
			# will be applied to each map. But this assumes that the
			# maps all see the same signal. So this basically just
			# shows the CMB-relevant signal. To see what would happen
			# to a, the signal we actually care about, we should use
			# P instead
			print i, self.Q[i][0,0]
			sumPHCH += self.Q[i]*self.B[i]*Hmean[i]**2*self.iN[i]*self.Q[i][0,0]
			sumHCHB += self.B[i]*Hmean[i]**2*self.iN[i]*self.Q[i][0,0]
			core  += Hmean[i]**2*self.iN[i]*self.B[i]**2
		core  = self.mapset.S/(1+self.mapset.S*core)
		term2 = PHCHB*sumHCHB*core
		res = sumPHCH - term2
		return res
	def sim(self):
		# Debug: make noise and signal simulations for each of our maps
		s = map_ifft(self.mapset.S**0.5*map_fft(enmap.rand_gauss(self.shape, self.wcs))).astype(self.dtype)
		n = enmap.samewcs([self.H[i]**-1*map_ifft(self.hN[i]**-1*map_fft(enmap.rand_gauss(self.shape, self.wcs))).astype(self.dtype) for i in range(self.nmap)],s)
		n += s
		return n

def spec2var(spec_2d):
	return np.mean(spec_2d)**0.5

def blockvar(m, bsize=10):
	return enmap.downgrade(m**2, bsize)

class Coadder:
	def __init__(self, mapset):
		self.mapset = mapset
		# Extract and flatten all our input maps
		self.m  = [split.data.map             for dataset in mapset.datasets for split in dataset.splits]
		self.H  = [split.data.H               for dataset in mapset.datasets for split in dataset.splits]
		self.iN = [dataset.iN                 for dataset in mapset.datasets for split in dataset.splits]
		self.B  = [dataset.beam_2d/mapset.target_beam for dataset in mapset.datasets for split in dataset.splits]
		self.shape, self.wcs = mapset.shape, mapset.wcs
		self.dtype= mapset.dtype
		self.ctype= np.result_type(self.dtype,0j)
		self.npix = self.shape[-2]*self.shape[-1]
		self.nmap = len(self.m)
		self.tot_div = enmap.zeros(self.shape[-2:], self.wcs, self.dtype)
		for H in self.H: self.tot_div += H**2
	def calc_rhs(self):
		# Calc rhs = B'HCH m
		rhs = enmap.zeros(self.shape, self.wcs, self.dtype)
		for i in range(self.nmap):
			rhs += map_ifft(self.B[i]*map_fft(self.H[i]*map_ifft(self.iN[i]*map_fft(self.H[i]*self.m[i]))))
		return rhs
	def calc_coadd(self, rhs, maxiter=250, cg_tol=1e-4, verbose=False, dump_dir=None):
		# solve (B'HCHB)x = rhs. For preconditioner, we will use the full-fourier approximation,
		# so M = (B'Hmean C Hmean B)". The solution itself is done in fourier space, to save
		# some ffts.
		def zip(map): return map.reshape(-1).view(self.dtype)
		def unzip(x): return enmap.ndmap(x.view(self.ctype).reshape(self.shape), self.wcs)
		def A(x):
			fmap = unzip(x)
			fres = enmap.zeros(self.shape, self.wcs, self.ctype)
			for i in range(self.nmap):
				fres += self.B[i]*map_fft(self.H[i]*map_ifft(self.iN[i]*map_fft(self.H[i]*map_ifft(self.B[i]*fmap))))
			return zip(fres)
		prec = enmap.zeros(self.shape, self.wcs, self.ctype)
		for i in range(self.nmap):
			Hmean = np.mean(self.H[i])
			prec += Hmean**2*self.B[i]**2*self.iN[i]
		prec = 1/(prec + np.max(prec)*1e-8)
		def M(x): return zip(prec*unzip(x))
		solver = cg.CG(A, zip(map_fft(rhs)), M=M)
		for i in range(maxiter):
			t1 = time.time()
			solver.step()
			t2 = time.time()
			if verbose:
				print "%5d %15.7e %5.2f" % (solver.i, solver.err, t2-t1)
			if dump_dir is not None and solver.i in [1,2,5,10,20,50] + range(100,10000,100):
				enmap.write_map(dump_dir + "/map_step%04d.fits" % solver.i, map_ifft(unzip(solver.x)))
			if solver.err < cg_tol: break
		if dump_dir is not None:
			enmap.write_map(dump_dir + "/map_final.fits", map_ifft(unzip(solver.x)))
		return map_ifft(unzip(solver.x))

class SourceSZFinder:
	"""Identify point sources and tSZ clusters in the map set given by mapset.
	Fits a position and amplitude per frequency for point sources, and
	a position and scale for clusters. Works by first applying a matched
	filter and looking for strong candidates. These are then fit and subtracted
	from the maps, after which a new pass is performed with a lower threshold.
	
	We model the data as m = Pa + ntot = BQa + ntot. This looks similar to the
	model in the filter, but differs in that a is now no longer a map that's
	in common between all the data maps, but instead a [namp] lenght vector of
	the number of free amplitudes. This would be namp = 1 for sz, namp = nfreq
	for normal point sources and namp = ndataset for variable point sources.

	P is therefore [npix*nmap,namp], and can be factorized as P = BQ,
	with B = block_diag(B1,B2,...), and Q = [npix*nmap,namp], being simply
	the signal template for each degree of freedom that each map sees.

	We model all the splits inside a dataset as being equivalent, so we can
	reduce everything to one map and noise matrix etc. per dataset to begin with.
	I'll assume that has been done in the following.

	chiqs = (m-BQa)'Ntot"(M-BQa) = a'Q'B'Ntot"BQa - 2a'Q'B'Ntot"m + m'Ntot"m
	Can precompute U=B'Ntot"B (block-diagonal) and v=B'Ntot"m (map stack).
	To do this we need Ntot = (N + BbS'b'B), where b' = [I I I ...] is a
	bradcasting matrix.
	B Ntot" B = BN"B - BN"Bb(S" + b'BN"Bb)"b'BN"B
	B Ntot" B is huge due to the outer products in the last term. But we don't
	need the explicit matrix. We can precompute BN"B and (S" + b'BN"B')" = core,
	and use them directly in the likelihood:

	chisq = c1 + c2 + c3
	c1 = a'Q'BN"BQa - (a'QBN"Bb) core (a'QBN"Bb)', which does not involve large matrices
	c2 = -2a'Q'BN"m + 2(a'QBN"Bb) core (m'N"Bb)',  likewise
	c3 = m'N"m - (m'N"Bb) core (m'N"Bb)',          likewise, and also constant

	When sampling a, we get a <- N(ahat, A), where
	A" = Q'B Ntot" BQ = Q'BN"BQ - (Q'BN"Bb) core (Q'BN"Bb)' (tiny, [namp,namp])
	ahat = A Q'B Ntot" m = A[Q'BN"m - (Q'BN"Bb) core (Q'BN"m)'

	Can write Q = pos * profile, where profile encodes only the shape (identity matrix
	for point source), and pos only encodes the position. pos would be a kronecker delta
	for pixel-center positions. For off-center positions, it would be fourier-shifted.

	What should the final output from this class be? Aside from position and shape for
	each source/cluster, we also want amplitudes. But for point sources amplitudes
	can change form map to map, and those are useful numbers to measure. So how about
	outputting an overall amp, a per-freq amp and a per-map amplitude? The problem with
	this is that different tiles will have different maps going into them, so when
	combining the catalogues we will have incompatible amplitude vectors. Could handle
	this by adding a map-id to the output. But really, tile issues are not the responsibility
	of this class, so we can worry about elsewhere.
	"""
	def __init__(self, mapset, sz_scales=[0.1,0.5,2.0,4.0], snmin=4, npass=2, spix=33, mode="auto", ignore_mean=True):
		self.mapset = mapset
		self.scales = sz_scales
		self.snmin  = snmin
		self.npass  = npass
		self.spix   = spix
		self.npix  = spix*spix
		# Precompute the matrix building blocks. These are common for all the sources
		self.C = [np.linalg.inv(ps2d_to_mat(1/d.iN.preflat[0],spix).reshape(self.npix,self.npix)) for d in mapset.datasets]
		self.B = [ps2d_to_mat(d.beam_2d,spix).reshape(self.npix,self.npix)             for d in mapset.datasets]
		self.S = ps2d_to_mat(mapset.S,spix).reshape(self.npix, self.npix)
		# Eigenvalues have too large span for standard inverse. Should be safe for C though.
		self.iS = utils.eigpow(self.S, -1)
		self.freqs  = [d.freq for d in mapset.datasets]
		self.nfreq  = len(self.freqs)
		self.nmap   = len(mapset.datasets)
		self.mode   = mode
		self.ignore_mean = ignore_mean
		# This is used for the model subtraction
		self.pixshape = enmap.pixshape(mapset.shape, mapset.wcs)/utils.arcmin
		self.pixshape[1] *= np.cos(enmap.pix2sky(mapset.shape, mapset.wcs, [mapset.shape[-2]/2,mapset.shape[-1]/2])[0])
	def analyze(self, npass=None, verbosity=0):
		"""Iterator that repeatedly look for sources and clusters, subtract the mean model from the maps,
		and look again. This is be able to find objects close to other, strong objects. Yields a
		bunch(catalogue, snmaps, model) each time."""
		if npass is None: npass = self.npass
		for it in range(npass):
			info = self.single_pass(verbosity=verbosity)
			self.subtract_model(info.model_full)
			info.i = it
			yield info
	def subtract_model(self, model):
		"""Subtract a model (a map per dataset) from our mapset. This modifies the
		mapset that was used to build this object. The same model is used across the
		maps in a split."""
		for i, d in enumerate(self.mapset.datasets):
			for split in d.splits:
				split.data.map.preflat[0] -= model[i]
	def single_pass(self, snmin=None, verbosity=0):
		"""Performs a single pass of the full search. Returns a catalogue of statistics
		in the form of a numpy structured array with the fields
		"""
		if snmin is None: snmin = self.snmin
		cands, snmaps = self.find_candidates(snmin, maps=True)
		cands = prune_candidates(cands)
		model = enmap.zeros((self.nmap,)+self.mapset.shape[-2:], self.mapset.wcs, self.mapset.dtype)
		model_full = model*0
		# Set up the output source info array
		Nf = "%df" % self.nmap
		cattype = [
				("type","S10"),("pos","2f"),("dpos","2f"),("pos0","2f"),("fwhm","f"), ("dfwhm","f"), # spatial parameters
				("sn","f"),("sn0","f"),("amp","f"),("damp","f"),                      # overall strength
				("amps",Nf),("damps",Nf),                                             # individual amplitudes
				("npix","i"),                                                         # misc
			]
		cat = np.recarray(len(cands), cattype)
		# Evaluate each candidate
		for ci, cand in enumerate(cands):
			ipix  = utils.nint(cand.pix)
			lik   = self.get_likelihood(ipix, cand.type)
			ml    = lik.maximize(verbose = verbosity>=2)
			stats = lik.explore(ml.x, verbose = verbosity>=2, nsamp=50)
			# I used to just expand the mean models from the stats here, but
			# those only cover a thumbnail, and the sz wings can be wider than that.
			# So instead build a full-map model.
			#model      += expand_thumb(stats.model,      ipix, model.shape)
			#model_full += expand_thumb(stats.model_full, ipix, model.shape)
			if    cand.type == "sz":
				profile_shape = sz_map_profile(self.mapset.shape[-2:], self.mapset.wcs, fwhm=ml.x[2])
				profile_amps  = [sz_freq_core(freq*1e9) for freq in self.freqs]
			elif  cand.type == "ptsrc":
				profile_shape = model[0]*0
				profile_shape[0,0] = 1
				profile_amps  = [1 for freq in self.freqs]
			else: raise ValueError("Unknown object type '%s'" % cand.type)
			profile_shape = fft.shift(profile_shape, cand.pix)
			for di, d in enumerate(self.mapset.datasets):
				profile = map_ifft(d.beam_2d*map_fft(profile_shape))*profile_amps[di]
				model_full[di] += ml.amps_full.val[di]*profile
				model[di]      += ml.amps.val[lik.groups[di]]*profile
			# Populate catalogue
			c = cat[ci]
			# Spatial parameters
			c.type = cand.type
			c.pos  = enmap.pix2sky(self.mapset.shape, self.mapset.wcs, ipix+stats.x[:2])[::-1]/utils.degree # radec
			c.dpos = (np.diag(stats.x_cov)[:2]**0.5*enmap.pixshape(self.mapset.shape, self.mapset.wcs))[::-1]/utils.degree
			c.pos0 = cand.pos[::-1]/utils.degree
			if cand.type == "sz":
				c.fwhm  = stats.x[2]
				c.dfwhm = stats.x_cov[2,2]**0.5
			else: c.fwhm, c.dfwhm = 0, 0
			# Overall strength
			iA_ml  = utils.eigpow(ml.amps.cov, -1)
			c.sn   = (ml.amps.val.dot(iA_ml).dot(ml.amps.val))**0.5 # Conditional on ML position
			c.sn0  = cand.sn
			iA     = utils.eigpow(stats.amps.cov, -1)
			c.amp  = np.sum(iA.dot(stats.amps.val))/np.sum(iA)
			c.damp = np.sum(iA)**-0.5
			c.amps = stats.amps_full.val
			c.damps= np.diag(stats.amps_full.cov)**0.5
			# misc
			c.npix = cand.npix
			if verbosity >= 1:
				print "%3d %s" % (ci+1, format_catalogue(c)),
				sys.stdout.flush()
		# And return lots of useful stuff
		res = bunch.Bunch(catalogue = cat, snmaps = snmaps, model = model, model_full= model_full)
		return res
	def get_likelihood(self, pix, type, scale=0.5, mode=None):
		"""Return an object that can be used to evaluate the likelihood for a source
		near the given position, of the given type (ptsrc or sz). scale is
		an initial guess at the sz length scale."""
		ipix = utils.nint(pix)
		# Geometry of this slice
		cy, cx = ipix - self.spix//2
		shape, wcs = enmap.slice_geometry(self.mapset.shape[-2:], self.mapset.wcs,
				(slice(cy,cy+self.spix),slice(cx,cx+self.spix)))
		# 1. Build thumbs for each dataset
		rhs, iN = [], []
		for di, d in enumerate(self.mapset.datasets):
			dset_rhs  = np.zeros(self.npix)
			dset_iN   = np.zeros([self.npix, self.npix])
			for si, s in enumerate(d.splits):
				split_m  = np.asarray(extract_thumb(s.data.map.preflat[0], ipix, self.spix).reshape(-1))
				split_H  = np.asarray(extract_thumb(s.data.H, ipix, self.spix).reshape(-1))
				split_iN = np.asarray(split_H[:,None]*self.C[di]*split_H[None,:])
				if self.ignore_mean:
					# Make ourselves insensitive to the mean
					mvec = np.full(self.npix, 1.0/self.npix, split_iN.dtype)
					split_iN = project_out(split_iN, mvec)
				dset_rhs += split_iN.dot(split_m)
				dset_iN  += split_iN
			rhs.append(dset_rhs)
			iN.append(dset_iN)
		nmap = len(rhs)
		# Set up degree of freedom grouping
		if   mode is None:     mode = self.mode
		if   mode == "auto":   mode = "single" if type == "sz" else "perfreq"
		if   mode == "single":  groups = np.full(nmap, 0, int)
		elif mode == "perfreq": groups = np.unique(self.freqs, return_inverse=True)[1]
		elif mode == "permap":  groups = np.arange(nmap)
		else: raise ValueError("Unknown DOF mode '%s'" % mode)
		# 2. We need to know which
		if   type == "ptsrc":
			return PtsrcLikelihood(rhs, iN, self.B, self.iS, shape, wcs, groups=groups)
		elif type == "sz":
			return SZLikelihood(rhs, iN, self.B, self.iS, shape, wcs, self.freqs, groups=groups)
		else:
			raise ValueError("Unknown signal type '%s'" % type)
	def find_candidates(self, lim=5.0, maps=False):
		"""Find matched filter point source and sz candidates with S/N of at least lim.
		Returns a single list containing both ptsrc and sz candidates, sorted by S/N. They
		are returned as a recarray with the fields [sn, type, pos[2], pix[2], npix[2]].
		If maps=True, then the S/N maps that were used in the search will returned as a
		second argument [(type,map),(type,map),...]"""
		dtype  = [("sn","f"),("type","S10"),("pos","2f"),("pix","2f"),("npix","i")]
		cands  = []
		snmaps = []
		filter = SignalFilter(self.mapset)
		rhs    = filter.calc_rhs()
		mu     = filter.calc_mu(rhs)
		for name in ["ptsrc", "sz"]:
			submaps = []
			if name == "ptsrc":
				setup_profiles_ptsrc(self.mapset)
				alpha  = filter.calc_alpha(mu)
				dalpha = filter.calc_dalpha_empirical(alpha)
				snmap  = div_nonan(alpha, dalpha)
			elif name == "sz":
				snmap = None
				for scale in self.scales:
					setup_profiles_sz(self.mapset, scale)
					alpha  = filter.calc_alpha(mu)
					dalpha = filter.calc_dalpha_empirical(alpha)
					snmap_1scale = div_nonan(alpha, dalpha)
					if snmap is None: snmap = snmap_1scale
					else: snmap = np.maximum(snmap, snmap_1scale)
					submaps.append(bunch.Bunch(name="%03.1f"%scale, snmap=snmap_1scale))
			else: raise ValueError("Unknown signal type '%s'" % name)
			cand = find_candidates(snmap, lim, edge=self.mapset.apod_edge)
			cand.type  = name
			cands.append(cand)
			snmaps.append((name,snmap))
		cands = np.rec.array(np.concatenate(cands))
		cands = cands[np.argsort(cands.sn)[::-1]]
		if maps: return cands, snmaps
		else:    return cands

# What do I want to be able to do with the Likelihood object?
# I want to be able to find the ML, statistics and model.
# But the number of nonlinear parameters (parameters that need to
# be sampled over) varies, as does their meaning. It will also
# vary which amplitudes are considered to be free. If the sampling
# and maximization is done by an external object, then I must present
# common interface for the nonlinear and linear degrees of freedom.
#
# What about n_nonlin, n_lin, n_map, eval(nonlin) -> {posterior, lik, prior, nonlin, lin, amps, model}?
# Here nonlin = [y,x,...], lin is the amplitude for the independent groups in the fit, and
# amps is what the amplitudes would have been if they were not grouped.

class PtsrcLikelihood:
	def __init__(self, rhs, iN, B, iS, shape, wcs, groups=None, rmax=None):
		self.nmap  = len(rhs)
		self.npix  = len(rhs[0])
		self.shape, self.wcs = shape, wcs
		self.dtype = rhs[0].dtype
		self.rhs = rhs
		self.iN, self.B, self.iS = iN, B, iS
		# Compute core = S" + B'N"B and b'B'N"m

		#for i, r in enumerate(rhs):
		#	r = enmap.ndmap(rhs[i].reshape(shape),wcs)
		#	m = enmap.ndmap(np.linalg.solve(self.iN[i], rhs[i]).reshape(shape), wcs)
		#	enmap.write_map("test_map_%02d.fits" % i, m)
		#	enmap.write_map("test_rhs_%02d.fits" % i, r)
		#	E, V = np.linalg.eigh(self.iN[i])

		self.core  = iS.copy()
		self.bBiNm = np.zeros(self.npix, self.dtype)
		self.chisq = 0
		for i in range(self.nmap):
			self.core  += B[i].T.dot(iN[i]).dot(B[i])
			self.bBiNm += B[i].T.dot(rhs[i])
			self.chisq += rhs[i].dot(np.linalg.solve(iN[i],rhs[i]))
		self.icore = utils.eigpow(self.core, -1)
		#self.chisq -= self.bBiNm.T.dot(np.linalg.solve(self.core, self.bBiNm))
		self.chisq -= self.bBiNm.T.dot(self.icore).dot(self.bBiNm)
		# Set up our position vector
		self.pos_base = np.zeros(shape)
		self.pos_base[tuple([n//2 for n in shape])] = 1
		# Set up the groups of linear parameters that vary together.
		# These take the form of an array mapping nmap -> ngroup
		self.groups = groups if groups is not None else np.arange(self.nmap)
		# The interface we expose for samplers
		self.nlin = np.max(self.groups)+1
		self.namp = self.nmap
		self.nx   = 2
		self.x0   = np.zeros(2)
		# Our position prior, in pixels
		self.rmax = rmax if rmax is not None else min(*shape)/2
		self.xscale = np.array([self.rmax,self.rmax])
	def __call__(self, x):
		t1 = time.time()
		Q = self.calc_Q(x)
		iA_full, arhs_full = self.calc_amp_eqsys(Q)
		# Find the ML amplitudes for all the amplitudes
		A_full    = utils.eigpow(iA_full, -1)
		ahat_full = A_full.dot(arhs_full)
		# Do the same for our groups
		iA   = binmat(iA_full,   self.groups)
		arhs = binvec(arhs_full, self.groups)
		A    = utils.eigpow(iA, -1)
		ahat = A.dot(arhs)
		# Get the amp-marginalized -2log-likelihood (including Jeffrey's prior, excluding constants)
		# This is negative due to how the marginalization works out
		lik   = self.chisq-ahat.dot(arhs)
		prior = self.calc_prior(x, ahat, A)
		post  = lik + prior
		# Get our model. Computing this every step is not necessary, but it
		# only accounts for 0.3% of the time this function takes
		def get_model(a):
			model = np.array([self.B[i].dot(Q[i]*a[i]) for i in range(self.nmap)])
			return enmap.ndmap(model.reshape((-1,)+self.shape), self.wcs)
		model_full = get_model(ahat_full)
		model      = get_model(ahat[self.groups])
		t2 = time.time()
		return bunch.Bunch(
			posterior  = post,
			likelihood = lik,
			prior      = prior,
			x          = x,
			amps       = bunch.Bunch(val = ahat,      cov = A),
			amps_full  = bunch.Bunch(val = ahat_full, cov = A_full),
			model      = model,
			model_full = model_full,
			t          = t2-t1,
		)
	def calc_prior(self, x, a, A):
		r = np.sum(x[:2]**2)**0.5
		if r > self.rmax: return np.inf
		res = -2*log_prob_gauss_positive(a,A)
		return res
	def calc_Q(self, x):
		pos_mat = shift_nonperiodic(self.pos_base, x[:2]).reshape(self.npix)
		return [pos_mat]*self.nmap
	def calc_amp_eqsys(self, Q):
		"""Compute the the left and right hand side of the fit amplitude
		iA and arhs such that iA*ahat = arhs, where ahat is the ML estimator
		for the amplitudes and iA is their inverse covariance. This is
		done for the case where the point source is located at the given
		position pos = [dy,dx] in (non-integer) pixels as measured from the
		center of the region. returns (iA, arhs, P). These are all that is
		needed both to
		1. compute the ML amplitudes
		2. sample from the amplitude distribution
		3. compute the fit log-posterior, assuming a Jeffrey's prior
		4. reduce to a smaller number of amplitude degrees of freedom
		5. compute the model
		"""
		t1 = time.time()
		# A"   = Q'B' Ntot" BQ = Q'B'N"BQ - (Q'B'N"Bb) core" (Q'B'N"Bb)'
		# ahat = A Q'B' Ntot" m = A[Q'BN"m - (Q'B'N"Bb) core" b'B'N"m
		# So we need Q'B'N"Bb, Q'B'N"BQ, Q'B'N"m and b'B'N"m
		# How does b work in this case? b broadcasts from a single
		# map to all maps: it is b' = [I I I I].
		QBNBQ = np.zeros([self.nmap, self.nmap])
		QBNBb = np.zeros([self.nmap, self.npix], self.dtype)
		QBNm  = np.zeros(self.nmap, self.dtype)
		# Everything is block-diagonal, except core, which we have already handled
		for i in range(self.nmap):
			QB = Q[i].dot(self.B[i].T)
			QBNBQ[i,i] = QB.dot(self.iN[i]).dot(QB.T)
			QBNBb[i]   = QB.dot(self.iN[i]).dot(self.B[i])
			# At least for planck, the mean value of rhs (and presumably of m)
			# is given a large weight here, even though the CMB should suppress
			# those scales. Should investigate why.
			QBNm[i]    = QB.dot(self.rhs[i])
			# This is the (untransposed) pointing/profile matrix. Useful for evaluating the model
		iA    = QBNBQ - QBNBb.dot(self.icore).dot(QBNBb.T)
		arhs  = QBNm  - QBNBb.dot(self.icore).dot(self.bBiNm)
		#iA    = QBNBQ - QBNBb.dot(np.linalg.solve(self.core, QBNBb.T))
		#arhs  = QBNm  - QBNBb.dot(np.linalg.solve(self.core, self.bBiNm))
		return iA, arhs
	def maximize(self, verbose=False):
		self.i = 0
		def f(x):
			res = self(x)
			if verbose:
				print "%3d %s" % (self.i, self.format_sample(res))
				sys.stdout.flush()
			self.i += 1
			return res.posterior
		x = optimize.fmin_powell(f, self.x0, disp=False)
		return self(x)
	def explore(self, x0=None, nsamp=50, fburn=0.5, fwalker=1.0, stepscale=2.0, verbose=False):
		# Sample the likelihood using metropolis, and gather statistics.
		# For the position this is simple, but for the amplitudes each
		# sample is a distribution. To get the mean and cov of this,
		# we can imagine sampling a bunch of amps for each step.
		# <a> = mean_{bunch,samp} a_{bunch,samp} = mean_bunch mean_samp a_{bunch,samp} = mean(a_ml)
		# cov(a) = <aa'> = mean_bunch mean_samp (a_ml_bunch + da_bunch - mean_ml)*(...)'
		# cov(a_ml) + mean(A)
		# To keep thing simple, we will not measure the pos-amp cross terms
		# We will sample using emcee, to avoid having to worry about rescaling the step size
		if x0 is None: x0 = self.x0
		nburn   = utils.nint(nsamp*fburn)
		# Set up the initial walkers
		nwalker = utils.nint((self.nx+1)*fwalker)
		init_dx = self.xscale*0.01
		points, vals = [], []
		for i in range(10000):
			x = x0 + np.random.standard_normal(self.nx)*init_dx
			v = self(x)
			if np.isfinite(v.posterior):
				points.append(x)
				vals.append(v)
			if len(points) >= nwalker:
				break
		points = np.array(points)

		#points  = np.array([x0 + np.random.standard_normal(self.nx)*init_dx for i in range(nwalker)])
		#vals    = [self(x) for x in points]

		# Set up our output statistics
		# Nonlinear parameters
		mean_x  = np.zeros(self.nx)
		mean_xx = np.zeros([self.nx,self.nx])
		# Main amplitudes
		mean_a  = np.zeros(self.nlin)
		mean_aa = np.zeros([self.nlin, self.nlin])
		mean_A  = np.zeros([self.nlin, self.nlin])
		# Full amplitudes
		mean_a_full  = np.zeros(self.namp)
		mean_aa_full = np.zeros([self.namp, self.namp])
		mean_A_full  = np.zeros([self.namp, self.namp])
		# Models
		mean_model = vals[0].model*0
		mean_model_full = vals[0].model_full*0
		# Time spent
		mean_t = 0
		nsum = 0

		# Loop over samples
		for si in range(-nburn, nsamp):
			next_points = points.copy()
			next_vals   = list(vals)
			# Try making a step with each walker
			for wi in range(nwalker):
				oi = np.random.randint(nwalker-1)
				if oi == wi: oi += 1
				stretch  = draw_emcee_stretch(stepscale)
				cand_pos = points[oi] + stretch*(points[wi]-points[oi])
				cand_val = self(cand_pos)
				p_accept = stepscale**(self.nx-1)*np.exp(0.5*(vals[wi].posterior - cand_val.posterior))
				r = np.random.uniform(0,1)
				if r < p_accept:
					next_points[wi] = cand_pos
					next_vals[wi]   = cand_val
				x, v = next_points[wi], next_vals[wi]
				if verbose:
					print "%3d %d %s" % (si, wi, self.format_sample(v))
					sys.stdout.flush()
				# Accumulate statistics if we're done with burnin
				if si >= 0:
					mean_x  += x
					mean_xx += x[:,None]*x[None,:]
					mean_a  += v.amps.val
					mean_aa += v.amps.val[:,None]*v.amps.val[None,:]
					mean_A  += v.amps.cov
					mean_a_full  += v.amps_full.val
					mean_aa_full += v.amps_full.val[:,None]*v.amps_full.val[None,:]
					mean_A_full  += v.amps_full.cov
					mean_model      += v.model
					mean_model_full += v.model_full
					mean_t += v.t
					nsum += 1
			# Done with all walkers. Update current state
			points = next_points
			vals   = next_vals
	
		arrs = [mean_x, mean_xx, mean_a, mean_aa, mean_A, mean_a_full, mean_aa_full, mean_A_full, mean_model, mean_model_full]
		for arr in arrs: arr /= nsum
		mean_t /= nsum

		res = bunch.Bunch(
				x      = mean_x,
				x_cov  = mean_xx - mean_x[:,None]*mean_x[None,:],
				amps   = bunch.Bunch(
					val = mean_a, cov = mean_aa - mean_a[:,None]*mean_a[None,:] + mean_A),
				amps_full = bunch.Bunch(
					val = mean_a_full, cov = mean_aa_full - mean_a_full[:,None]*mean_a_full[None,:] + mean_A_full),
				model = mean_model,
				model_full = mean_model_full,
				t = mean_t,
				# these don't really make sense, but include them to make this a fully valid sample
				prior = 0,
				likelihood = 0,
				posterior = 0,
			)
		return res
	@staticmethod
	def format_sample(sample):
		nx  = len(sample.x)
		dx  = np.diag(sample.x_cov)**0.5 if "x_cov" in sample else sample.x*0
		res = ""
		for i in range(nx):
			res += " %6.3f %6.3f " % (sample.x[i], dx[i])
		res += " t %6.3f L %8.2f amp" % (sample.t, sample.posterior)
		for i, a in enumerate(sample.amps_full.val):
			da = sample.amps_full.cov[i,i]**0.5
			res += "  %6.3f %6.3f" % (a/1e3, da/1e3)
		return res

class SZLikelihood(PtsrcLikelihood):
	def __init__(self, rhs, iN, B, iS, shape, wcs, freqs, groups=None, rmax=None, smax=None, smin=None):
		PtsrcLikelihood.__init__(self, rhs, iN, B, iS, shape, wcs, groups=groups, rmax=rmax)
		self.smax = smax if smax is not None else 10
		self.smin = smin if smin is not None else 0.2
		self.x0   = [0,0,0.5]
		self.xscale = np.array([self.rmax,self.rmax,self.smax])
		self.nx   = 3
		self.freqs= freqs
		# Needed for sz P evaluation
		self.pixshape = enmap.pixshape(shape, wcs)/utils.arcmin
		self.pixshape[1] *= np.cos(enmap.pix2sky(shape, wcs, [shape[-2]/2,shape[-1]/2])[0])
	def calc_prior(self, x, a, A):
		r  = np.sum(x[:2]**2)**0.5
		if   r > self.rmax:    return np.inf
		elif x[2] <= 0:        return np.inf
		res = -2*log_prob_gauss_positive(a,A)
		if   x[2] < self.smin: res += np.exp((self.smin/x[2]-1)*10)-1
		elif x[2] > self.smax: res += np.exp((x[2]/self.smax-1)*10)-1
		return res
	def calc_Q(self, x):
		pos, scale = x[:2], max(x[2],1e-2)
		sz_prof  = sz_2d_profile(self.shape, self.pixshape, pos=pos, fwhm=scale).reshape(-1)
		## Get the distance from our chosen pixel position to all the other pixels
		#dists    = np.sum(((pos[:,None,None] - self.pixmap)*self.pixshape[:,None,None])**2,0)**0.5/utils.arcmin
		#sz_prof  = self.sz_fun(dists, scale)/scale*1e-2
		#sz_prof  = enmap.downgrade(sz_prof, self.nsub).reshape(-1)
		cache = {}
		Q = []
		for freq in self.freqs:
			if freq not in cache:
				cache[freq] = sz_prof * sz_freq_core(freq*1e9)
			Q.append(cache[freq])
		#print "Q"
		#sys.stdout.flush()
		#np.savetxt("/dev/stdout", Q[0].reshape(self.shape)[::3,::3]*100, fmt="%7.2f")
		#sys.stdout.flush()
		return Q


class SourceSZFinder2:
	def __init__(self, mapset, sz_scales=[0.1,0.5,1.0,2.0], snmin=4, npass=4, pass_snmin=6, spix=33, mode="auto", ignore_mean=True, nmax=None, model_snmin=5):
		self.mapset = mapset
		self.scales = sz_scales
		self.snmin  = snmin
		self.npass  = npass
		self.pass_snmin = pass_snmin
		self.spix   = spix
		self.npix  = spix*spix
		# Precompute the matrix building blocks. These are common for all the sources
		self.C = [np.linalg.inv(ps2d_to_mat(1/d.iN.preflat[0],spix).reshape(self.npix,self.npix)) for d in mapset.datasets]
		self.B = [ps2d_to_mat(d.beam_2d,spix).reshape(self.npix,self.npix)             for d in mapset.datasets]
		self.S = ps2d_to_mat(mapset.S,spix).reshape(self.npix, self.npix)
		# Eigenvalues have too large span for standard inverse. Should be safe for C though.
		self.iS = utils.eigpow(self.S, -1)
		self.freqs  = [d.freq for d in mapset.datasets]
		self.nfreq  = len(self.freqs)
		self.nmap   = len(mapset.datasets)
		self.mode   = mode
		self.ignore_mean = ignore_mean
		# This is used for the model subtraction
		self.pixshape = enmap.pixshape(mapset.shape, mapset.wcs)/utils.arcmin
		self.pixshape[1] *= np.cos(enmap.pix2sky(mapset.shape, mapset.wcs, [mapset.shape[-2]/2,mapset.shape[-1]/2])[0])
		# min H level to avoid degenerate matrices
		self.h_tol = 1e-5
		self.h_min = 1e-10
		self.nmax  = nmax
		self.model_snmin = model_snmin
	def analyze(self, npass=None, verbosity=0):
		"""Loop through all analysis passes, returning a final bunch(catalogue, snmaps, model).
		The catalogue will be the union of all the individual stage catalogues, and the model
		will be the sums. The snmap returned will be the initial SNmap. If more fine grained
		information is needed, use the multi_pass iterator."""
		cats, snmapss, models = [], [], []
		for info in self.multi_pass(npass=npass, verbosity=verbosity):
			cats.append(info.catalogue)
			snmapss.append(info.snmaps)
			models.append(info.model)
		catalogue = np.concatenate(cats)
		catalogue = np.rec.array(catalogue[np.argsort(catalogue["sn"])[::-1]])
		snmaps    = snmapss[0]
		snresid   = snmapss[-1]
		model     = enmap.samewcs(np.sum(models,0),models[0])
		return bunch.Bunch(catalogue=catalogue, snmaps=snmaps, snresid=snresid, model=model)
	def multi_pass(self, npass=None, verbosity=0, nmax=None):
		"""Iterator that repeatedly look for sources and clusters, subtract the mean model from the maps,
		and look again. This is be able to find objects close to other, strong objects. Yields a
		bunch(catalogue, snmaps, model) each time."""
		if npass is None: npass = self.npass
		others = None
		for it in range(npass):
			if verbosity >= 1: print "Pass %d" % (it+1)
			cands, snmaps = self.find_candidates(self.snmin, maps=True, verbosity=verbosity, others=others)
			others        = np.rec.array(np.concatenate([others,cands])) if others is not None else cands
			info          = self.measure_candidates(cands, verbosity=verbosity)
			info.snmaps   = snmaps
			self.subtract_model(info.model_full)
			info.i = it
			yield info
			# Stop iterating once we have found everything
			if np.sum(cands.sn > self.pass_snmin) == 0: break
	def subtract_model(self, model):
		"""Subtract a model (a map per dataset) from our mapset. This modifies the
		mapset that was used to build this object. The same model is used across the
		maps in a split."""
		for i, d in enumerate(self.mapset.datasets):
			for split in d.splits:
				split.data.map.preflat[0] -= model[i]
	def measure_candidates(self, cands, verbosity=0):
		"""Performs a single pass of the full search. Returns a catalogue of statistics
		in the form of a numpy structured array with the fields
		"""
		model       = enmap.zeros((self.nmap,)+self.mapset.shape[-2:], self.mapset.wcs, self.mapset.dtype)
		model_full  = model*0
		# Set up the output source info array
		cattype = self.get_catalogue_format(self.nmap)
		cat = np.recarray(len(cands), cattype)
		t0 = time.time()
		# Evaluate each candidate
		for ci, cand in enumerate(cands):
			ipix  = utils.nint(cand.pix)
			t1    = time.time()
			lik   = self.get_likelihood(ipix, cand.type)
			t2    = time.time()
			ml    = lik.maximize(verbose = verbosity>=3)
			t3    = time.time()
			if    cand.type == "sz":
				profile_shape = sz_map_profile(self.mapset.shape[-2:], self.mapset.wcs, fwhm=ml.x[2])
				profile_amps  = [sz_freq_core(freq*1e9) for freq in self.freqs]
			elif  cand.type == "ptsrc":
				profile_shape = model[0]*0
				profile_shape[0,0] = 1
				profile_amps  = [1 for freq in self.freqs]
			else: raise ValueError("Unknown object type '%s'" % cand.type)
			aoff = 3 if cand.type == "sz" else 2
			profile_shape = fft.shift(profile_shape, cand.pix)
			for di, d in enumerate(self.mapset.datasets):
				profile    = map_ifft(d.beam_2d*map_fft(profile_shape))*profile_amps[di]
				model_full[di] += ml.x[aoff:][lik.groups[di]]*profile
				if ml.sn > self.model_snmin:
					model[di] += ml.x[aoff:][lik.groups[di]]*profile
			# Populate catalogue
			c = cat[ci]
			# Spatial parameters
			c.type = cand.type
			c.pos  = enmap.pix2sky(self.mapset.shape, self.mapset.wcs, ipix+ml.x[:2])[::-1]/utils.degree # radec
			c.dpos = (np.diag(ml.x_cov)[:2]**0.5*enmap.pixshape(self.mapset.shape, self.mapset.wcs))[::-1]/utils.degree
			c.pos0 = cand.pos[::-1]/utils.degree
			if cand.type == "sz":
				c.fwhm  = ml.x[2]
				c.dfwhm = ml.x_cov[2,2]**0.5
			else: c.fwhm, c.dfwhm = 0, 0
			# Overall strength
			c.sn   = ml.sn
			c.sn0  = cand.sn
			iA     = utils.eigpow(ml.x_cov[aoff:,aoff:],-1)
			iAtot  = np.sum(iA)
			c.amp  = np.sum(iA.dot(ml.x[aoff:]))/iAtot
			c.damp = iAtot**-0.5
			c.amps = ml.afull
			c.damps= np.diag(ml.afull_cov)**0.5
			# misc
			c.npix = cand.npix
			t4 = time.time()
			if verbosity >= 2:
				print "%3d %4.1f %4.1f %s" % (ci+1, t2-t1, t3-t2, format_catalogue(c)),
				sys.stdout.flush()
		t5 = time.time()
		if verbosity >= 1:
			print "Measured %2d objects in %5.1f s" % (len(cands), t5-t0)
		# And return lots of useful stuff
		res = bunch.Bunch(catalogue = cat, model = model, model_full=model_full)
		return res
	def get_likelihood(self, pix, type, scale=0.5, mode=None):
		"""Return an object that can be used to evaluate the likelihood for a source
		near the given position, of the given type (ptsrc or sz). scale is
		an initial guess at the sz length scale."""
		ipix = utils.nint(pix)
		# Geometry of this slice
		cy, cx = ipix - self.spix//2
		shape, wcs = enmap.slice_geometry(self.mapset.shape[-2:], self.mapset.wcs,
				(slice(cy,cy+self.spix),slice(cx,cx+self.spix)))
		# 1. Build thumbs for each dataset
		rhs, iN = [], []
		for di, d in enumerate(self.mapset.datasets):
			dset_rhs  = np.zeros(self.npix)
			dset_iN   = np.zeros([self.npix, self.npix])
			for si, s in enumerate(d.splits):
				split_m  = np.asarray(extract_thumb(s.data.map.preflat[0], ipix, self.spix).reshape(-1))
				split_H  = np.asarray(extract_thumb(s.data.H, ipix, self.spix).reshape(-1))
				split_H  = np.maximum(split_H, max(self.h_min,np.max(split_H)*self.h_tol))
				split_iN = np.asarray(split_H[:,None]*self.C[di]*split_H[None,:])
				if self.ignore_mean:
					# Make ourselves (almost) insensitive to the mean. We don't fully remove it
					# to avoid zero eigenvalues
					mvec = np.full(self.npix, 1.0/self.npix, split_iN.dtype)
					split_iN = project_out(split_iN, mvec, frac=1-1e-5)
				dset_rhs += split_iN.dot(split_m)
				dset_iN  += split_iN
			rhs.append(dset_rhs)
			iN.append(dset_iN)
		m = [np.linalg.solve(iN[i], rhs[i]) for i in range(len(rhs))]
		nmap = len(rhs)
		# Set up degree of freedom grouping
		if   mode is None:     mode = self.mode
		if   mode == "auto":   mode = "single" if type == "sz" else "perfreq"
		if   mode == "single":  groups = np.full(nmap, 0, int)
		elif mode == "perfreq": groups = np.unique(self.freqs, return_inverse=True)[1]
		elif mode == "permap":  groups = np.arange(nmap)
		else: raise ValueError("Unknown DOF mode '%s'" % mode)
		# 2. We need to know which
		if   type == "ptsrc":
			return PtsrcLikelihood2(m, iN, self.B, self.iS, shape, wcs, groups=groups)
		elif type == "sz":
			return SZLikelihood2(m, iN, self.B, self.iS, shape, wcs, self.freqs, groups=groups)
		else:
			raise ValueError("Unknown signal type '%s'" % type)
	def find_candidates(self, lim=5.0, maps=False, prune=True, verbosity=0, others=None):
		"""Find matched filter point source and sz candidates with S/N of at least lim.
		Returns a single list containing both ptsrc and sz candidates, sorted by S/N. They
		are returned as a recarray with the fields [sn, type, pos[2], pix[2], npix[2]].
		If maps=True, then the S/N maps that were used in the search will returned as a
		second argument [(type,map),(type,map),...]"""
		dtype  = [("sn","f"),("type","S10"),("pos","2f"),("pix","2f"),("npix","i")]
		cands  = []
		snmaps = []
		t1     = time.time()
		filter = SignalFilter(self.mapset)
		rhs    = filter.calc_rhs()
		mu     = filter.calc_mu(rhs)
		print "find candidates"
		for name in ["ptsrc", "sz"]:
			submaps = []
			if name == "ptsrc":
				setup_profiles_ptsrc(self.mapset)
				alpha  = filter.calc_alpha(mu)
				dalpha = filter.calc_dalpha_empirical(alpha)
				snmap  = div_nonan(alpha, dalpha)
			elif name == "sz":
				snmap = None
				for si, scale in enumerate(self.scales):
					setup_profiles_sz(self.mapset, scale)
					alpha  = filter.calc_alpha(mu)
					dalpha = filter.calc_dalpha_empirical(alpha)
					snmap_1scale = div_nonan(alpha, dalpha)
					if snmap is None: snmap = snmap_1scale
					else: snmap = np.maximum(snmap, snmap_1scale)
					submaps.append(bunch.Bunch(name="%03.1f"%scale, snmap=snmap_1scale))
			else: raise ValueError("Unknown signal type '%s'" % name)
			cand = find_candidates(snmap, lim, edge=self.mapset.apod_edge)
			cand.type  = name
			cands.append(cand)
			snmaps.append((name,snmap))
		cands = np.rec.array(np.concatenate(cands))
		cands = cands[np.argsort(cands.sn)[::-1]]
		if self.nmax is not None:
			cands = cands[:self.nmax]
		if prune:
			cands = prune_candidates(cands, others=others, verbose=verbosity>=2)
		t2 = time.time()
		if verbosity >= 1:
			print "Found %3d candidates in %5.1f s" % (len(cands),t2-t1)
		if maps: return cands, snmaps
		else:    return cands
	@staticmethod
	def get_catalogue_format(nmap):
		Nf = "%df" % nmap
		return [
				("type","S10"),("pos","2f"),("dpos","2f"),("pos0","2f"),("fwhm","f"), ("dfwhm","f"), # spatial parameters
				("sn","f"),("sn0","f"),("amp","f"),("damp","f"),                      # overall strength
				("amps",Nf),("damps",Nf),                                             # individual amplitudes
				("npix","i"),                                                         # misc
			]

class PtsrcLikelihood2:
	def __init__(self, m, iN, B, iS, shape, wcs, groups=None, rmax=None):
		self.nmap  = len(m)
		self.npix  = len(m[0])
		self.shape, self.wcs = shape, wcs
		self.dtype = m[0].dtype
		self.m, self.iN, self.B, self.iS = np.asarray(m), iN, B, iS
		# Compute core = S" + B'N"B and b'B'N"m
		self.core   = iS.copy()
		for i in range(self.nmap):
			self.core  += B[i].T.dot(iN[i].dot(B[i]))
		self.icore  = utils.eigpow(self.core, -1)
		self.iNtotm = self.mul_iNtot(self.m)
		# Set up our position vector
		self.pos_base = np.zeros(shape)
		self.pos_base[tuple([n//2 for n in shape])] = 1
		# Set up the groups of linear parameters that vary together.
		# These take the form of an array mapping nmap -> ngroup
		self.groups = groups if groups is not None else np.arange(self.nmap)
		self.aoff   = 2
		self.nparam = self.aoff+np.max(self.groups)+1
		# Our position prior, in pixels
		self.rmax = rmax if rmax is not None else min(*shape)/2
		self.scale= np.array([1]*self.aoff+[1000]*(self.nparam-self.aoff))
		# Optimization
		self.cache = {}
	# These explore the full nonlinear + linear posterior. They take an x that is [nnon+nlin]
	def calc_log_posterior(self, x):
		"""Compute the minus log-posterior distribution. The log-likelihood part of this is,
		ignoring constant terms mlogL = 0.5*(d-m)'Ntot"(d-m)
		"""
		r     = self.m - self.calc_model(x)
		iNr   = self.get_cache("iNr", x, lambda: self.mul_iNtot(r))
		logL  = 0.5*np.sum(r*iNr)
		logL += self.calc_log_prior(x)
		return logL
	def calc_dlog_posterior(self, x):
		"""Compute the derivate of the minus log-posterior, d mlogL  = -dm'Ntot"(d-m)"""
		iNr   = self.get_cache("iNr", x, lambda: self.mul_iNtot(self.m - self.calc_model(x)))
		# We now need the derivative of the model with respect to each parameter.
		# This includes dm/dpos, dm/dscale, dm/damp. The simplest is dm/damp,
		# since the model is just proportional to to the amplitude
		dm  = self.calc_dmodel(x)
		dlogL = np.zeros(self.nparam, self.dtype)
		for i in range(self.nparam):
			dlogL[i] = -np.sum(dm[i]*iNr)
		dlogL += self.calc_dlog_prior(x)
		return dlogL
	def calc_dlog_posterior_num(self, x, delta=1e-4):
		res = np.zeros(len(x))
		for i in range(len(x)):
			x1 = x.copy(); x1[i] -= delta
			y1 = self.calc_log_posterior(x1)
			x2 = x.copy(); x2[i] += delta
			y2 = self.calc_log_posterior(x2)
			res[i] = (y2-y1)/(2*delta)
		return res
	def calc_ddlog_posterior_num(self, x, delta=1e-6):
		res = np.zeros([len(x),len(x)])
		for i in range(len(x)):
			d  = delta*self.scale[i]
			x1 = x.copy(); x1[i] -= d
			y1 = self.calc_dlog_posterior(x1)
			x2 = x.copy(); x2[i] += d
			y2 = self.calc_dlog_posterior(x2)
			res[i] = (y2-y1)/(2*d)
		res = 0.5*(res+res.T) # symmetrize
		return res
	def calc_log_prior(self, x):
		r     = (1e-10+np.sum(x[:2]**2))**0.5
		logp  = soft_prior(r, self.rmax)
		logp += np.sum(soft_prior(-x[2:], 0))
		return logp
	def calc_dlog_prior(self, x):
		r     = (1e-10+np.sum(x[:2]**2))**0.5
		dlogp = np.zeros(len(x))
		dlogp[:2] = x[:2]/r*soft_prior(r, self.rmax, deriv=True)
		dlogp[2:] = -soft_prior(-x[2:], 0, deriv=True)
		return dlogp
	def calc_initial_value(self):
		x0 = np.zeros(self.nparam)
		iA, arhs = self.calc_amp_eqsys(x0)
		iA   = binmat(iA,   self.groups)
		arhs = binvec(arhs, self.groups)
		# Our prior requires positive amplitudes
		x0[2:] = np.abs(np.linalg.solve(iA, arhs))
		return x0
	def calc_amp_eqsys(self, x):
		"""Compute the left and right hand side of the conditional amplitude
		distribution, returning iA, arhs. The P that is passed represents the
		*non-zero blocks* of the response matrix. This means that we can't just
		multiply P by matrices as if it were a full matrix.
		
		iA   = P'N"P, ahat = iA"P'N"P
		"""
		P    = self.calc_P(x)
		iNP  = self.get_cache("iNP", x, lambda: np.array([self.iN[i].dot(P[i]) for i in range(self.nmap)]))
		PNP  = np.zeros([self.nmap, self.nmap], self.dtype)
		PNB  = np.zeros([self.nmap, self.npix], self.dtype)
		# Everything is block-diagonal, except core, which we have already handled
		for i in range(self.nmap):
			PNP[i,i] = P[i].dot(iNP[i])
			PNB[i]   = self.B[i].dot(iNP[i]) # B and iN are symmetric
		iA    = PNP - PNB.dot(self.icore.dot(PNB.T))
		arhs  = np.sum(P*self.iNtotm,1)
		return iA, arhs
	#def calc_damp_eqsys(self, x):
	#	P    = self.calc_P(x)
	#	iNP  = self.get_cache("iNP", x, lambda: np.array([self.iN[i].dot(P[i]) for i in range(self.nmap)]))
	#	diA  = np.zeros([2, self.nmap, self.nmap], self.dtype)
	#	darhs= np.zeros([2, self.nmap], self.dtype)
	#	for c in range(2):
	#		dP   = self.calc_P(x, pos_deriv=c)
	#		dPNd = np.zeros([self.nmap, self.npix], self.dtype)
	#		dPNP = np.zeros([self.nmap, self.nmap], self.dtype)
	#		dPNB = np.zeros([self.nmap, self.npix], self.dtype)
	#		PNB  = np.zeros([self.nmap, self.npix], self.dtype)
	#		for i in range(self.nmap):
	#			dPNP[i,i]  = dP[i].dot(iNP[i])
	#			PNB[i]     = self.B[i].dot(iNP[i])
	#			dPNB[i]    = self.B[i].dot(self.iN[i].dot(dP[i]))
	#		diA[c]  = dPNP - dPNB.dot(self.icore.dot(PNB.T))
	#		diA[c] += diA[c].T
	#		dPNd    = dP.dot(self.iNtotm)
	#		darhs   = 
	def calc_model(self, x):
		amps = x[self.aoff:][self.groups]
		P    = self.calc_P(x)
		model= np.zeros([self.nmap, self.npix], self.dtype)
		for i in range(self.nmap):
			model[i] = amps[i]*P[i]
		return model
	def calc_dmodel_num(self, x, delta=1e-3):
		res = np.zeros([len(x),self.nmap,self.npix],self.dtype)
		for i in range(len(x)):
			x1 = x.copy(); x1[i] -= delta
			y1 = self.calc_model(x1)
			x2 = x.copy(); x2[i] += delta
			y2 = self.calc_model(x2)
			res[i] = (y2-y1)/(2*delta)
		return res
	def calc_dmodel(self, x):
		"""Calculate the derivative of the model with respect to all the parameters.
		This will result in nmap*nparam maps, but the maps are tiny, so this should be fine."""
		pos, amp = x[:2], x[2:]
		dmodel = np.zeros([self.nparam, self.nmap, self.npix], self.dtype)
		# First get the derivative by position
		for i in range(2):
			dP = self.calc_P(x, pos_deriv=i)
			for gi, g in enumerate(self.groups):
				dmodel[i,gi] = amp[g]*dP[gi]
		# Then get the derivative by amplitude
		x_noamp     = x.copy()
		x_noamp[2:] = 1.0
		unit_model  = self.calc_model(x_noamp)
		for gi, g in enumerate(self.groups):
			dmodel[2+g,gi] = unit_model[gi]
		return dmodel
	def calc_Q(self, x, pos_deriv=None):
		# Use periodic shifting to avoid pixel-border derivative problems
		pos_mat = fft.shift(self.pos_base, x[:2], deriv=pos_deriv).reshape(self.npix)
		return [pos_mat]*self.nmap
	def calc_P(self, x, pos_deriv=None):
		"""Returns the matrix P that takes us from our full amplitudes
		to the full model. While P is logically [nmap*npix,nmap], the amplitudes do not
		mix, so it is enough to return the [nmap,npix] non-zero entries.
		P_full[map1*pix,map2] = P_small[map2,pix]*delta(map1,map2).
		"""
		return self.get_cache("P",
				np.concatenate([x[:self.aoff],[-1 if pos_deriv is None else pos_deriv]]),
				lambda: np.array([self.B[i].dot(Q) for i,Q in enumerate(
					self.calc_Q(x, pos_deriv=pos_deriv))]))
	def mul_iNtot(self, m):
		"""Ntot" = N" - N"Bb(S" + b'B'N"Bb)"b'B'N" = N" - N"Bb icore b'BN"""
		iNm   = np.zeros([self.nmap, self.npix], self.dtype)
		cbBiN = np.zeros(self.npix, self.dtype)
		for i in range(self.nmap):
			iNm[i] = self.iN[i].dot(m[i])
			cbBiN  += self.B[i].dot(iNm[i])
		cbBiN = self.icore.dot(cbBiN)
		for i in range(self.nmap):
			iNm[i] -= self.iN[i].dot(self.B[i].dot(cbBiN))
		return iNm
	def get_cache(self, key, x, f):
		if key not in self.cache or not np.allclose(x, self.cache[key][0], rtol=1e-14, atol=0):
			self.cache[key] = [np.array(x), np.array(f())]
		return self.cache[key][1].copy()
	def format_sample(self, x):
		return "%8.3f %8.3f" % tuple(x[:2]) + " %8.3f"*(len(x)-2)%tuple(x[2:]/1e3)
	def maximize(self, x0=None, verbose=False):
		"""Find the maximum likelihood point, along with a fisher error estimate."""
		if x0 is None:
			x0 = self.calc_initial_value()
		self.n  = 0
		def f(x):
			self.n += 1
			t1 = time.time()
			p = self.calc_log_posterior(x)
			t2 = time.time()
			if verbose:
				print "%3d %5.2f %9.3f %s" % (self.n, t2-t1, p, self.format_sample(x))
				sys.stdout.flush()
			return p
		x, logP, dlogP, ihess, nf, ng, w = optimize.fmin_bfgs(f, x0, self.calc_dlog_posterior, disp=False, full_output=True)
		# The returned inverse hessian is not very reliable. Use
		# numerical derivative instead
		ddlogP = self.calc_ddlog_posterior_num(x)
		x_cov  = np.linalg.inv(ddlogP)

		# Get the full amps too
		iA, arhs = self.calc_amp_eqsys(x)
		A = utils.eigpow(iA, -1)
		a = A.dot(arhs)
		# Estimate the equivalent S/N
		xtmp = x.copy(); xtmp[self.aoff:] = 0
		logP_null = self.calc_log_posterior(xtmp)
		sn = max(2*(logP_null-logP)-self.nparam,0)**0.5
		# Get the gaussian errors
		return bunch.Bunch(x=x, x_cov=x_cov, sn=sn, logP=logP, logP_null=logP, afull=a, afull_cov=A)

class SZLikelihood2(PtsrcLikelihood2):
	def __init__(self, m, iN, B, iS, shape, wcs, freqs, groups=None, rmax=None, smax=None, smin=None):
		PtsrcLikelihood2.__init__(self, m, iN, B, iS, shape, wcs, groups=groups, rmax=rmax)
		self.smax   = smax if smax is not None else 10
		self.smin   = smin if smin is not None else 0.2
		self.aoff   = 3
		self.nparam = self.aoff+np.max(self.groups)+1
		self.freqs  = freqs
		# Needed for sz P evaluation
		self.pixshape = enmap.pixshape(shape, wcs)/utils.arcmin
		self.pixshape[1] *= np.cos(enmap.pix2sky(shape, wcs, [shape[-2]/2,shape[-1]/2])[0])
		self.scale= np.array([1]*self.aoff+[1000]*(self.nparam-self.aoff))
	def calc_initial_value(self):
		x0 = np.zeros(self.nparam)
		x0[2] = 1.0
		iA, arhs = self.calc_amp_eqsys(x0)
		iA       = binmat(iA,   self.groups)
		arhs     = binvec(arhs, self.groups)
		# We require positive amplitudes
		x0[3:]   = np.abs(np.linalg.solve(iA, arhs))
		return x0
	def calc_log_prior(self, x):
		r     = (1e-10+np.sum(x[:2]**2))**0.5
		logp  = soft_prior(r, self.rmax)
		logp += soft_prior(-x[2], -self.smin) + soft_prior(x[2], self.smax)
		logp += np.sum(soft_prior(-x[3:], 0))
		return logp
	def calc_dlog_prior(self, x):
		r     = (1e-10+np.sum(x[:2]**2))**0.5
		dlogp = np.zeros(len(x))
		dlogp[:2] = x[:2]/r*soft_prior(r, self.rmax, deriv=True)
		dlogp[2]  = -soft_prior(-x[2], -self.smin, deriv=True) + soft_prior(x[2], self.smax, deriv=True)
		dlogp[3:] = -soft_prior(-x[3:], 0, deriv=True)
		return dlogp
	def calc_Q(self, x, pos_deriv=None, scale_deriv=False):
		pos, scale = x[:2], max(x[2],1e-2)
		# Use periodic shifting to avoid pixel-border derivative problems
		sz_prof  = sz_2d_profile(self.shape, self.pixshape, pos=pos, fwhm=scale,
				pos_deriv=pos_deriv, scale_deriv=scale_deriv, periodic=True).reshape(-1)
		cache = {}
		Q = []
		for freq in self.freqs:
			if freq not in cache:
				cache[freq] = sz_prof * sz_freq_core(freq*1e9)
			Q.append(cache[freq])
		return Q
	def calc_P(self, x, pos_deriv=None, scale_deriv=False):
		return self.get_cache("P",
				np.concatenate([x[:self.aoff],[-1 if pos_deriv is None else pos_deriv, scale_deriv]]),
				lambda: np.array([self.B[i].dot(Q) for i,Q in enumerate(
					self.calc_Q(x, pos_deriv=pos_deriv, scale_deriv=scale_deriv))]))
	def calc_dmodel(self, x):
		"""Calculate the derivative of the model with respect to all the parameters.
		This will result in nmap*nparam maps, but the maps are tiny, so this should be fine."""
		pos, scale, amp = x[:2], x[2], x[3:]
		dmodel = np.zeros([self.nparam, self.nmap, self.npix], self.dtype)
		# First get the derivative by position
		for i in range(2):
			dP = self.calc_P(x, pos_deriv=i)
			for gi, g in enumerate(self.groups):
				dmodel[i,gi] = amp[g]*dP[gi]
		# Then the derivative by scale
		dP = self.calc_P(x, scale_deriv=True)
		for gi, g in enumerate(self.groups):
			dmodel[2,gi] = amp[g]*dP[gi]
		# Then get the derivative by amplitude
		x_noamp     = x.copy()
		x_noamp[3:] = 1.0
		unit_model  = self.calc_model(x_noamp)
		for gi, g in enumerate(self.groups):
			dmodel[3+g,gi] = unit_model[gi]
		return dmodel
	def format_sample(self, x):
		return "%8.3f %8.3f %8.3f" % tuple(x[:3]) + " %8.3f"*(len(x)-3)%tuple(x[3:]/1e3)


# This verison uses both ML amps and derivatives at the same time
class SourceSZFinder3:
	def __init__(self, mapset, sz_scales=[0.1,0.25,0.5,1.0,2.0], snmin=4, npass=4, pass_snmin=6, spix=33, mode="auto", ignore_mean=True, nmax=None, model_snmin=5):
		print "A %8.3f %8.3f" % (memory.current()/1024.**3, memory.max()/1024.**3)
		self.mapset = mapset
		self.scales = sz_scales
		self.snmin  = snmin
		self.npass  = npass
		self.pass_snmin = pass_snmin
		self.spix   = spix
		self.npix   = spix*spix
		# Precompute the matrix building blocks. These are common for all the sources
		self.C = [np.linalg.inv(ps2d_to_mat(1/d.iN.preflat[0],spix).reshape(self.npix,self.npix)) for d in mapset.datasets]
		self.B = [ps2d_to_mat(d.beam_2d,spix).reshape(self.npix,self.npix)             for d in mapset.datasets]
		self.S = ps2d_to_mat(mapset.S,spix).reshape(self.npix, self.npix)
		# Eigenvalues have too large span for standard inverse. Should be safe for C though.
		self.iS = utils.eigpow(self.S, -1)
		self.freqs  = [d.freq for d in mapset.datasets]
		self.nfreq  = len(self.freqs)
		self.nmap   = len(mapset.datasets)
		self.mode   = mode
		self.ignore_mean = ignore_mean
		# This is used for the model subtraction
		self.pixshape = enmap.pixshape(mapset.shape, mapset.wcs)/utils.arcmin
		self.pixshape[1] *= np.cos(enmap.pix2sky(mapset.shape, mapset.wcs, [mapset.shape[-2]/2,mapset.shape[-1]/2])[0])
		# min H level to avoid degenerate matrices
		self.h_tol = 1e-5
		self.h_min = 1e-10
		self.nmax  = nmax
		self.model_snmin = model_snmin
		print "B %8.3f %8.3f" % (memory.current()/1024.**3, memory.max()/1024.**3)
	def analyze(self, npass=None, verbosity=0):
		"""Loop through all analysis passes, returning a final bunch(catalogue, snmaps, model).
		The catalogue will be the union of all the individual stage catalogues, and the model
		will be the sums. The snmap returned will be the initial SNmap. If more fine grained
		information is needed, use the multi_pass iterator."""
		cats, snmapss, models = [], [], []
		for info in self.multi_pass(npass=npass, verbosity=verbosity):
			cats.append(info.catalogue)
			snmapss.append(info.snmaps)
			models.append(info.model)
		catalogue = np.concatenate(cats)
		catalogue = np.rec.array(catalogue[np.argsort(catalogue["sn"])[::-1]])
		snmaps    = snmapss[0]
		snresid   = snmapss[-1]
		model     = enmap.samewcs(np.sum(models,0),models[0])
		return bunch.Bunch(catalogue=catalogue, snmaps=snmaps, snresid=snresid, model=model)
	def multi_pass(self, npass=None, verbosity=0, nmax=None):
		"""Iterator that repeatedly look for sources and clusters, subtract the mean model from the maps,
		and look again. This is be able to find objects close to other, strong objects. Yields a
		bunch(catalogue, snmaps, model) each time."""
		if npass is None: npass = self.npass
		others = None
		for it in range(npass):
			if verbosity >= 1: print "Pass %d" % (it+1)
			# We gradually restrict our edge as we iterate to reduce the effect of ringing
			# from objects that are just ourside our edge, and hence can't be properly subtracted
			cands, snmaps = self.find_candidates(self.snmin, maps=True, verbosity=verbosity, others=others, edge=self.mapset.apod_edge*it)
			others        = np.rec.array(np.concatenate([others,cands])) if others is not None else cands
			info          = self.measure_candidates(cands, verbosity=verbosity)
			info.snmaps   = snmaps
			self.subtract_model(info.model_full)
			info.i = it
			yield info
			# Stop iterating once we have found everything
			if np.sum(cands.sn > self.pass_snmin) == 0: break
	def subtract_model(self, model):
		"""Subtract a model (a map per dataset) from our mapset. This modifies the
		mapset that was used to build this object. The same model is used across the
		maps in a split."""
		for i, d in enumerate(self.mapset.datasets):
			for split in d.splits:
				split.data.map.preflat[0] -= model[i]
	def measure_candidates(self, cands, verbosity=0):
		"""Performs a single pass of the full search. Returns a catalogue of statistics
		in the form of a numpy structured array with the fields
		"""
		model       = enmap.zeros((self.nmap,)+self.mapset.shape[-2:], self.mapset.wcs, self.mapset.dtype)
		model_full  = model*0
		# Set up the output source info array
		cattype = self.get_catalogue_format(self.nmap)
		cat = np.recarray(len(cands), cattype)
		t0 = time.time()
		# Evaluate each candidate
		for ci, cand in enumerate(cands):
			ipix  = utils.nint(cand.pix)
			print "C %8.3f %8.3f" % (memory.current()/1024.**3, memory.max()/1024.**3)
			t1    = time.time()
			lik   = self.get_likelihood(ipix, cand.type)
			print "D %8.3f %8.3f" % (memory.current()/1024.**3, memory.max()/1024.**3)
			t2    = time.time()
			# First find the ML-point, which will be our parameter estimate
			ml    = lik.maximize(verbose = verbosity>=3)
			# Then sample the likelihood to get an error estimate. Fall back to
			# full likelihood exploration if the fisher matrix estimate fails
			print "E %8.3f %8.3f" % (memory.current()/1024.**3, memory.max()/1024.**3)
			stats = lik.fisher(ml.x)
			if np.any(np.linalg.eigh(stats.x_cov)[0] <= 0):
				stats = lik.explore(ml.x, verbose=verbosity >=3)
			print "F %8.3f %8.3f" % (memory.current()/1024.**3, memory.max()/1024.**3)
			t3    = time.time()
			# Build ML model
			if    cand.type == "sz":
				profile_shape = sz_map_profile(self.mapset.shape[-2:], self.mapset.wcs, fwhm=ml.x[2])
				profile_amps  = [sz_freq_core(freq*1e9) for freq in self.freqs]
			elif  cand.type == "ptsrc":
				profile_shape = model[0]*0
				profile_shape[0,0] = 1
				profile_amps  = [1 for freq in self.freqs]
			else: raise ValueError("Unknown object type '%s'" % cand.type)
			profile_shape = fft.shift(profile_shape, cand.pix)
			for di, d in enumerate(self.mapset.datasets):
				profile    = map_ifft(d.beam_2d*map_fft(profile_shape))*profile_amps[di]
				model_full[di] += ml.a.val[lik.groups[di]]*profile
				if ml.sn > self.model_snmin:
					model[di] += ml.a.val[lik.groups[di]]*profile
			# Populate catalogue
			c = cat[ci]
			# Spatial parameters
			c.type = cand.type
			c.pos  = enmap.pix2sky(self.mapset.shape, self.mapset.wcs, ipix+ml.x[:2])[::-1]/utils.degree # radec
			c.dpos = (np.diag(stats.x_cov)[:2]**0.5*enmap.pixshape(self.mapset.shape, self.mapset.wcs))[::-1]/utils.degree
			c.pos0 = cand.pos[::-1]/utils.degree
			if cand.type == "sz":
				c.fwhm  = ml.x[2]
				c.dfwhm = stats.x_cov[2,2]**0.5
			else: c.fwhm, c.dfwhm = 0, 0
			# Overall strength
			c.sn   = ml.sn
			c.sn0  = cand.sn
			iAtot  = np.sum(ml.a.icov)
			c.amp  = np.sum(ml.a.rhs)/iAtot
			c.damp = iAtot**-0.5
			c.amps = ml.a_full.val
			c.damps= np.diag(ml.a_full.cov)**0.5
			# misc
			c.npix = cand.npix
			t4 = time.time()
			if verbosity >= 2:
				print "%3d %4.1f %4.1f %s" % (ci+1, t2-t1, t3-t2, format_catalogue(c)),
				sys.stdout.flush()
			print "F %8.3f %8.3f" % (memory.current()/1024.**3, memory.max()/1024.**3)
		t5 = time.time()
		if verbosity >= 1:
			print "Measured %2d objects in %5.1f s" % (len(cands), t5-t0)
		# And return lots of useful stuff
		res = bunch.Bunch(catalogue = cat, model = model, model_full=model_full)
		return res
	def get_likelihood(self, pix, type, scale=0.5, mode=None):
		"""Return an object that can be used to evaluate the likelihood for a source
		near the given position, of the given type (ptsrc or sz). scale is
		an initial guess at the sz length scale."""
		ipix = utils.nint(pix)
		# Geometry of this slice
		cy, cx = ipix - self.spix//2
		shape, wcs = enmap.slice_geometry(self.mapset.shape[-2:], self.mapset.wcs,
				(slice(cy,cy+self.spix),slice(cx,cx+self.spix)))
		# 1. Build thumbs for each dataset
		rhs, iN = [], []
		for di, d in enumerate(self.mapset.datasets):
			dset_rhs  = np.zeros(self.npix)
			dset_iN   = np.zeros([self.npix, self.npix])
			for si, s in enumerate(d.splits):
				split_m  = np.asarray(extract_thumb(s.data.map.preflat[0], ipix, self.spix).reshape(-1))
				split_H  = np.asarray(extract_thumb(s.data.H, ipix, self.spix).reshape(-1))
				split_H  = np.maximum(split_H, max(self.h_min,np.max(split_H)*self.h_tol))
				split_iN = np.asarray(split_H[:,None]*self.C[di]*split_H[None,:])
				if self.ignore_mean:
					# Make ourselves (almost) insensitive to the mean. We don't fully remove it
					# to avoid zero eigenvalues
					mvec = np.full(self.npix, 1.0/self.npix, split_iN.dtype)
					split_iN = project_out(split_iN, mvec, frac=1-1e-5)
				dset_rhs += split_iN.dot(split_m)
				dset_iN  += split_iN
			rhs.append(dset_rhs)
			iN.append(dset_iN)
		nmap = len(rhs)
		m = [np.linalg.solve(iN[i], rhs[i]) for i in range(nmap)]
		del rhs
		# Set up degree of freedom grouping
		if   mode is None:     mode = self.mode
		if   mode == "auto":   mode = "single" if type == "sz" else "perfreq"
		if   mode == "single":  groups = np.full(nmap, 0, int)
		elif mode == "perfreq": groups = np.unique(self.freqs, return_inverse=True)[1]
		elif mode == "permap":  groups = np.arange(nmap)
		else: raise ValueError("Unknown DOF mode '%s'" % mode)
		# 2. We need to know which
		if   type == "ptsrc":
			return PtsrcLikelihood3(m, iN, self.B, self.iS, shape, wcs, groups=groups)
		elif type == "sz":
			return SZLikelihood3(m, iN, self.B, self.iS, shape, wcs, self.freqs, groups=groups)
		else:
			raise ValueError("Unknown signal type '%s'" % type)
	def find_candidates(self, lim=5.0, maps=False, prune=True, verbosity=0, others=None, edge=0):
		"""Find matched filter point source and sz candidates with S/N of at least lim.
		Returns a single list containing both ptsrc and sz candidates, sorted by S/N. They
		are returned as a recarray with the fields [sn, type, pos[2], pix[2], npix[2]].
		If maps=True, then the S/N maps that were used in the search will returned as a
		second argument [(type,map),(type,map),...]"""
		dtype  = [("sn","f"),("type","S10"),("pos","2f"),("pix","2f"),("npix","i")]
		cands  = []
		snmaps = []
		t1     = time.time()
		filter = SignalFilter(self.mapset)
		rhs    = filter.calc_rhs()
		mu     = filter.calc_mu(rhs, verbose=verbosity >= 3)
		print "find candidates"
		#enmap.write_map("test_mu.fits", map_ifft(enmap.enmap(mu, mu[0].wcs)))
		#1/0
		for name in ["ptsrc", "sz"]:
			submaps = []
			print name
			if name == "ptsrc":
				setup_profiles_ptsrc(self.mapset)
				alpha  = filter.calc_alpha(mu)
				dalpha = filter.calc_dalpha_empirical(alpha)
				snmap  = div_nonan(alpha, dalpha)
			elif name == "sz":
				snmap = None
				for si, scale in enumerate(self.scales):
					print scale
					setup_profiles_sz(self.mapset, scale)
					print "calc alpha"
					alpha  = filter.calc_alpha(mu)
					print "calc_dalpha"
					dalpha = filter.calc_dalpha_empirical(alpha)
					snmap_1scale = div_nonan(alpha, dalpha)
					if snmap is None: snmap = snmap_1scale
					else: snmap = np.maximum(snmap, snmap_1scale)
					submaps.append(bunch.Bunch(name="%03.1f"%scale, snmap=snmap_1scale))
			else: raise ValueError("Unknown signal type '%s'" % name)
			cand = find_candidates(snmap, lim, edge=edge)
			cand.type  = name
			cands.append(cand)
			snmaps.append((name,snmap))
		cands = np.rec.array(np.concatenate(cands))
		cands = cands[np.argsort(cands.sn)[::-1]]
		if self.nmax is not None:
			cands = cands[:self.nmax]
		if prune:
			cands = prune_candidates(cands, others=others, verbose=verbosity>=2)
		t2 = time.time()
		if verbosity >= 1:
			print "Found %3d candidates in %5.1f s" % (len(cands),t2-t1)
		if maps: return cands, snmaps
		else:    return cands
	@staticmethod
	def get_catalogue_format(nmap):
		Nf = "%df" % nmap
		return [
				("type","S10"),("pos","2f"),("dpos","2f"),("pos0","2f"),("fwhm","f"), ("dfwhm","f"), # spatial parameters
				("sn","f"),("sn0","f"),("amp","f"),("damp","f"),                      # overall strength
				("amps",Nf),("damps",Nf),                                             # individual amplitudes
				("npix","i"),                                                         # misc
			]

class PtsrcLikelihood3:
	def __init__(self, m, iN, B, iS, shape, wcs, groups=None, rmax=None):
		self.nmap  = len(m)
		self.npix  = len(m[0])
		self.shape, self.wcs = shape, wcs
		self.dtype = m[0].dtype
		self.m, self.iN, self.B, self.iS = np.asarray(m), iN, B, iS
		# Compute core = S" + B'N"B and b'B'N"m
		self.core   = iS.copy()
		for i in range(self.nmap):
			self.core  += B[i].T.dot(iN[i].dot(B[i]))
		self.icore  = utils.eigpow(self.core, -1)
		self.iNtotm = self.mul_iNtot(self.m)
		# Set up our position vector
		self.pos_base = np.zeros(shape)
		self.pos_base[tuple([n//2 for n in shape])] = 1
		# Set up the groups of linear parameters that vary together.
		# These take the form of an array mapping nmap -> ngroup
		self.groups = groups if groups is not None else np.arange(self.nmap)
		self.nx   = 2
		self.nlin = np.max(self.groups)+1
		# Our position prior, in pixels
		self.rmax = rmax if rmax is not None else min(*shape)/2
		self.scale= np.array([1]*self.nx)
		# Optimization
		self.cache = {}
	@property
	def nparam(self): return self.nx + self.nlin
	# These explore the posterior with respect to the nonlinear parameters x
	def calc_log_posterior(self, x):
		"""Compute the minus log-posterior distribution. The log-likelihood part of this is,
		ignoring constant terms mlogL = 0.5*(d-m)'Ntot"(d-m) ~ -0.5*ahat'A"ahat = -0.5*arhs'A arhs.
		This should be evaluated after grouping in order to enforce our group prior.
		"""
		iA_full, arhs_full = self.calc_amp_eqsys(x)
		A_full   = utils.eigpow(iA_full, -1)
		ahat_full= A_full.dot(arhs_full)
		iA, arhs = binmat(iA_full, self.groups), binvec(arhs_full, self.groups)
		A        = utils.eigpow(iA, -1)
		ahat     = A.dot(arhs)
		logL  = -0.5*arhs.dot(ahat)
		# Prior
		logL += self.calc_log_prior(x, ahat, A)
		res = bunch.Bunch(
				x      = x,
				logL   = logL,
				a      = bunch.Bunch(val = ahat, cov = A, rhs=arhs, icov=iA),
				a_full = bunch.Bunch(val = ahat_full, cov=A_full, rhs=arhs_full, icov=iA_full)
			)
		return res
	def calc_log_prior(self, x, ahat, A):
		r     = (1e-10+np.sum(x[:2]**2))**0.5
		logp  = -log_prob_gauss_positive(ahat, A)
		logp += soft_prior(r, self.rmax)
		# Why was this one here?
		#logp += np.sum(soft_prior(-x[2:], 0))
		return logp
	def calc_initial_value(self):
		return np.zeros(self.nx, self.dtype)
	def calc_amp_eqsys(self, x):
		"""Compute the left and right hand side of the conditional amplitude
		distribution, returning iA, arhs.
		iA   = P'N"P, arhs = P'N"d, ahat = iA"arhs
		"""
		P    = self.calc_P(x)
		iNP  = self.get_cache("iNP", x, lambda: np.array([self.iN[i].dot(P[i]) for i in range(self.nmap)]))
		PNP  = np.zeros([self.nmap, self.nmap], self.dtype)
		PNB  = np.zeros([self.nmap, self.npix], self.dtype)
		# Everything is block-diagonal, except core, which we have already handled
		for i in range(self.nmap):
			PNP[i,i] = P[i].dot(iNP[i])
			PNB[i]   = self.B[i].dot(iNP[i]) # B and iN are symmetric
		iA    = PNP - PNB.dot(self.icore.dot(PNB.T))
		arhs  = np.sum(P*self.iNtotm,1)
		return iA, arhs
	def calc_Q(self, x, deriv=None):
		# Use periodic shifting to avoid pixel-border derivative problems
		pos_mat = fft.shift(self.pos_base, x, deriv=deriv).reshape(self.npix)
		return [pos_mat]*self.nmap
	def calc_P(self, x, deriv=None):
		"""Returns the matrix P that takes us from our full amplitudes
		to the full model. While P is logically [nmap*npix,nmap], the amplitudes do not
		mix, so it is enough to return the [nmap,npix] non-zero entries.
		P_full[map1*pix,map2] = P_small[map2,pix]*delta(map1,map2).
		"""
		return self.get_cache("P",
				np.concatenate([x,[-1 if deriv is None else deriv]]),
				lambda: np.array([self.B[i].dot(Q) for i,Q in enumerate(
					self.calc_Q(x, deriv=deriv))]))
	def mul_iNtot(self, m):
		"""Ntot" = N" - N"Bb(S" + b'B'N"Bb)"b'B'N" = N" - N"Bb icore b'BN"""
		iNm   = np.zeros([self.nmap, self.npix], self.dtype)
		cbBiN = np.zeros(self.npix, self.dtype)
		for i in range(self.nmap):
			iNm[i] = self.iN[i].dot(m[i])
			cbBiN  += self.B[i].dot(iNm[i])
		cbBiN = self.icore.dot(cbBiN)
		for i in range(self.nmap):
			iNm[i] -= self.iN[i].dot(self.B[i].dot(cbBiN))
		return iNm
	def get_cache(self, key, x, f):
		if key not in self.cache or not np.allclose(x, self.cache[key][0], rtol=1e-14, atol=0):
			self.cache[key] = [np.array(x), np.array(f())]
		return self.cache[key][1].copy()
	def maximize(self, x0=None, verbose=False):
		"""Find the maximum likelihood point."""
		if x0 is None:
			x0 = self.calc_initial_value()
		self.n  = 0
		def f(x):
			self.n += 1
			t1 = time.time()
			info = self.calc_log_posterior(x)
			t2 = time.time()
			if verbose:
				print "%3d %5.2f %9.3f %s" % (self.n, t2-t1, info.logL, self.format_sample(info))
				sys.stdout.flush()
			return info.logL
		x, logP, _, nit, nfun, warn = optimize.fmin_powell(f, x0, disp=False, full_output=True)
		# Get the full amps
		res    = self.calc_log_posterior(x)
		res.sn = np.max(res.a.val.dot(res.a.icov.dot(res.a.val))-self.nparam,0)**0.5
		return res
	def set_up_walkers(self, x0, nwalker):
		# Set up the initial walkers
		init_dx = self.scale*0.01
		points, vals = [], []
		# This ensures that all the walkers are valid
		for i in range(10000):
			x = x0 + np.random.standard_normal(self.nx)*init_dx
			info = self.calc_log_posterior(x)
			if np.isfinite(info.logL):
				points.append(x)
				vals.append(info)
			if len(points) >= nwalker:
				break
		return np.array(points), vals
	def explore(self, x0=None, nsamp=100, fburn=0.5, fwalker=1.0, stepscale=2.0, verbose=False):
		# Sample the likelihood using metropolis, and gather statistics.
		# For the position this is simple, but for the amplitudes each
		# sample is a distribution. To get the mean and cov of this,
		# we can imagine sampling a bunch of amps for each step.
		# <a> = mean_{bunch,samp} a_{bunch,samp} = mean_bunch mean_samp a_{bunch,samp} = mean(a_ml)
		# cov(a) = <aa'> = mean_bunch mean_samp (a_ml_bunch + da_bunch - mean_ml)*(...)'
		# cov(a_ml) + mean(A)
		# To keep thing simple, we will not measure the pos-amp cross terms
		# We will sample using emcee, to avoid having to worry about rescaling the step size
		if x0 is None: x0 = self.x0
		nburn   = utils.nint(nsamp*fburn)
		# Set up the initial walkers
		nwalker = utils.nint((self.nx+1)*fwalker)
		points, vals = self.set_up_walkers(x0, nwalker)
		# Set up our output statistics
		# Nonlinear parameters
		mean_x  = np.zeros(self.nx)
		mean_xx = np.zeros([self.nx,self.nx])
		# Main amplitudes
		mean_a  = np.zeros(self.nlin)
		mean_aa = np.zeros([self.nlin, self.nlin])
		mean_A  = np.zeros([self.nlin, self.nlin])
		# Full amplitudes
		mean_a_full  = np.zeros(self.nmap)
		mean_aa_full = np.zeros([self.nmap, self.nmap])
		mean_A_full  = np.zeros([self.nmap, self.nmap])
		nsum = 0
		# Loop over samples
		for si in range(-nburn, nsamp):
			next_points = points.copy()
			next_vals   = list(vals)
			# Try making a step with each walker
			for wi in range(nwalker):
				oi = np.random.randint(nwalker-1)
				if oi == wi: oi += 1
				stretch  = draw_emcee_stretch(stepscale)
				cand_pos = points[oi] + stretch*(points[wi]-points[oi])
				cand_val = self.calc_log_posterior(cand_pos)
				p_accept = stepscale**(self.nx-1)*np.exp(vals[wi].logL - cand_val.logL)
				r = np.random.uniform(0,1)
				if r < p_accept:
					next_points[wi] = cand_pos
					next_vals[wi]   = cand_val
				x, v = next_points[wi], next_vals[wi]
				if verbose:
					print "%3d %d %s" % (si, wi, self.format_sample(v))
					sys.stdout.flush()
				# Accumulate statistics if we're done with burnin
				if si >= 0:
					mean_x  += x
					mean_xx += x[:,None]*x[None,:]
					mean_a  += v.a.val
					mean_aa += v.a.val[:,None]*v.a.val[None,:]
					mean_A  += v.a.cov
					mean_a_full  += v.a_full.val
					mean_aa_full += v.a_full.val[:,None]*v.a_full.val[None,:]
					mean_A_full  += v.a_full.cov
					nsum += 1
			# Done with all walkers. Update current state
			points = next_points
			vals   = next_vals
	
		arrs = [mean_x, mean_xx, mean_a, mean_aa, mean_A, mean_a_full, mean_aa_full, mean_A_full]
		for arr in arrs: arr /= nsum

		res = bunch.Bunch(
				x = mean_x, x_cov = mean_xx - mean_x[:,None]*mean_x[None,:],
				a = bunch.Bunch(
					val = mean_a,
					cov = mean_aa - mean_a[:,None]*mean_a[None,:] + mean_A),
				a_full = bunch.Bunch(
					val = mean_a_full,
					cov = mean_aa_full - mean_a_full[:,None]*mean_a_full[None,:] + mean_A_full),
			)
		return res
	def fisher(self, x_ml, nper=3, step=1e-3):
		"""Compute fisher errors around x_ml, whichi should be the maximum-likelihood point"""
		t1 = time.time()
		res    = self.calc_log_posterior(x_ml)
		samp_x = x_ml + ((np.mgrid[(slice(0,nper),)*self.nx]-(nper-1)/2.0)*step).reshape(self.nx,-1).T
		samps  = [self.calc_log_posterior(x) for x in samp_x]
		samp_y = np.array([samp.logL for samp in samps])
		t2 = time.time()
		# We model the loglik as a quadratic form in dx. This has nx**2 (4 or 9) degrees of freedom,
		# which is fine since we have nper**nx (16 or 64) data points
		# logP = off + 0.5*(x-x0)'icov(x-x0)
		def zip(icov, x0, off): return np.concatenate([icov[np.triu_indices(len(icov))], [off]])
		def unzip(x):
			inds = np.triu_indices(self.nx)
			ntri = len(inds[0])
			icov = np.zeros([self.nx, self.nx], self.dtype)
			icov[inds] = x[:ntri]
			icov += np.tril(icov.T, 1)
			x0  = x_ml*0
			off = x[ntri]
			return icov, x0+x_ml, off
		def f(x):
			icov, x0, off = unzip(x)
			dx    = samp_x - x0
			model = np.sum(np.einsum("...i,ij->...j", dx, icov)*dx,1) + off
			resid = samp_y - model
			chisq = np.sum(resid**2)
			return chisq
		x0 = zip(np.diag(self.scale)*1e-3, x_ml, res.logL)
		x  = optimize.fmin_powell(f, x0, disp=False)
		icov, x0, off = unzip(x)
		# This gives us the nonlinear part. What about the linear amplitudes?
		# Just return ML for now
		res.x = x0
		res.x_cov = utils.eigpow(icov, -1)
		return res
	def format_sample(self, v):
		return " %8.3f"*len(v.x) % tuple(v.x) + " %8.3f"*len(v.a.val) % tuple(v.a.val/1e3)

class SZLikelihood3(PtsrcLikelihood3):
	def __init__(self, m, iN, B, iS, shape, wcs, freqs, groups=None, rmax=None, smin=None, smax=None):
		PtsrcLikelihood3.__init__(self, m, iN, B, iS, shape, wcs, groups=groups, rmax=rmax)
		self.freqs = freqs
		self.nx    = 3
		self.scale = np.array([1]*self.nx)
		self.smax  = smax if smax is not None else 10
		self.smin  = smin if smin is not None else 0.2
		# Needed for sz P evaluation
		self.pixshape = enmap.pixshape(shape, wcs)/utils.arcmin
		self.pixshape[1] *= np.cos(enmap.pix2sky(shape, wcs, [shape[-2]/2,shape[-1]/2])[0])
	def calc_initial_value(self):
		return np.array([0.0,0.0,1.0])
	def calc_log_prior(self, x, ahat, A):
		r     = (1e-10+np.sum(x[:2]**2))**0.5
		logp  = -log_prob_gauss_positive(ahat, A)
		logp += soft_prior(r, self.rmax)
		logp += soft_prior(-x[2], -self.smin) + soft_prior(x[2], self.smax)
		logp += np.sum(soft_prior(-x[2:], 0))
		return logp
	def calc_Q(self, x, deriv=None):
		pos, scale = x[:2], max(x[2],1e-2)
		# Use periodic shifting to avoid pixel-border derivative problems
		pos_deriv, scale_deriv = None, False
		if deriv is not None and deriv < 2:  pos_deriv = deriv
		if deriv is not None and deriv == 2: scale_deriv = True
		sz_prof  = sz_2d_profile(self.shape, self.pixshape, pos=pos, fwhm=scale,
				pos_deriv=pos_deriv, scale_deriv=scale_deriv, periodic=True).reshape(-1)
		cache = {}
		Q = []
		for freq in self.freqs:
			if freq not in cache:
				cache[freq] = sz_prof * sz_freq_core(freq*1e9)
			Q.append(cache[freq])
		return Q

#class PtsrcLikelihood3:
#	def __init__(self, m, iN, B, iS, shape, wcs, groups=None, rmax=None):
#		self.nmap  = len(m)
#		self.npix  = len(m[0])
#		self.shape, self.wcs = shape, wcs
#		self.dtype = m[0].dtype
#		self.m, self.iN, self.B, self.iS = np.asarray(m), iN, B, iS
#		# Compute core = S" + B'N"B and b'B'N"m
#		self.core   = iS.copy()
#		for i in range(self.nmap):
#			self.core  += B[i].T.dot(iN[i].dot(B[i]))
#		self.icore  = utils.eigpow(self.core, -1)
#		self.iNtotm = self.mul_iNtot(self.m)
#		# Set up our position vector
#		self.pos_base = np.zeros(shape)
#		self.pos_base[tuple([n//2 for n in shape])] = 1
#		# Set up the groups of linear parameters that vary together.
#		# These take the form of an array mapping nmap -> ngroup
#		self.groups = groups if groups is not None else np.arange(self.nmap)
#		self.nx   = 2
#		self.nlin = np.max(self.groups)+1
#		# Our position prior, in pixels
#		self.rmax = rmax if rmax is not None else min(*shape)/2
#		self.scale= np.array([1]*self.nx+[1000]*(self.nlin))
#		# Optimization
#		self.cache = {}
#	@property
#	def nparam(self): return self.nx + self.nlin
#	# These explore the posterior with respect to the nonlinear parameters x
#	def calc_log_posterior(self, x, return_info=False):
#		"""Compute the minus log-posterior distribution. The log-likelihood part of this is,
#		ignoring constant terms mlogL = 0.5*(d-m)'Ntot"(d-m) ~ -0.5*ahat'A"ahat = -0.5*arhs'A arhs.
#		This should be evaluated after grouping in order to enforce our group prior.
#		"""
#		iA, arhs = self.calc_amp_eqsys(x)
#		iA, arhs = binmat(iA, self.groups), binvec(arhs, self.groups)
#		A        = utils.eigpow(iA, -1)
#		ahat     = A.dot(arhs)
#		logL  = -0.5*arhs.dot(ahat)
#		# Amplitude prior
#		logL += -log_prob_gauss_positive(ahat, A)
#		# Other priors
#		logL += self.calc_log_prior(x)
#		if not return_info: return logL
#		else: return logL, bunch.Bunch(iA=iA, arhs=arhs, ahat=ahat)
#	def calc_dlog_posterior(self, x):
#		# We need d(arhs' A arhs) = 2*darhs A arhs - arhs' A diA A arhs = 2*darhs'ahat - ahat diA ahat
#		# darhs ahat + sym + ahat diA ahat
#		iA,  arhs  = self.calc_amp_eqsys(x)
#		diA, darhs = self.calc_damp_eqsys(x)
#		iA, arhs = binmat(iA, self.groups), binvec(arhs, self.groups)
#		diA   = np.array([binmat(diA[c],   self.groups) for c in range(self.nx)])
#		darhs = np.array([binvec(darhs[c], self.groups) for c in range(self.nx)])
#		try: ahat = np.linalg.solve(iA, arhs)
#		except np.linalg.LinAlgError: ahat = arhs*0
#		dlogL = np.zeros(self.nx, self.dtype)
#		for c in range(self.nx):
#			dlogL[c] = -0.5*(2*ahat.dot(darhs[c]) - ahat.dot(diA[c].dot(ahat)))
#		# Amplitude prior: dlogp/dx = dlogp/da*da/dx + dlogp/dA*dA/dx
#		# The latter is hard to evaluate analytically, since we end up needing
#		# the derivative of an eigenvalue decomposition
#
#		dlogL += self.calc_dlog_prior(x)
#		return dlogL
#	def calc_dlog_posterior_num(self, x, delta=1e-4):
#		res = np.zeros(len(x))
#		for i in range(len(x)):
#			x1 = x.copy(); x1[i] -= delta
#			y1 = self.calc_log_posterior(x1)
#			x2 = x.copy(); x2[i] += delta
#			y2 = self.calc_log_posterior(x2)
#			res[i] = (y2-y1)/(2*delta)
#		return res
#	def calc_ddlog_posterior_num(self, x, delta=1e-6):
#		res = np.zeros([len(x),len(x)])
#		for i in range(len(x)):
#			d  = delta*self.scale[i]
#			x1 = x.copy(); x1[i] -= d
#			y1 = self.calc_dlog_posterior(x1)
#			x2 = x.copy(); x2[i] += d
#			y2 = self.calc_dlog_posterior(x2)
#			res[i] = (y2-y1)/(2*d)
#		res = 0.5*(res+res.T) # symmetrize
#		return res
#	# We have multiple priors.
#	# 1. The bounds prior, which limits r etc. to have sensible values using
#	#    a soft cutoff.
#	# 2. The amplitude positivity prior, which is based on error functions
#	# 3. The source distribution prior, which will be a power law that prefers weak sources
#	# 4. The look elsewhere prior?
#	#
#	# Given our beam size we have Nn effective noise observations in an area A.
#	# Expected number of observations with amplitude b is Nb(b) = Nn*norm(b).
#	# Expected number of noise-free observations of true signal a in area a
#	# is Ns*powlaw(a). Expected number of noisy observations of amplitude b
#	# is Ns(b) = Ns*int powlaw(b-n)*norm(n) dn.
#	# Probability of being real is P(real|b) = Ns(b)/(Ns(b)+Nn(b)).
#	# And probability of having amplitude a if real is powlaw(a)*norm(b-a)
#	# This combines the noise term with the prior dietribution.
#	#
#	# When deriving the above I assumed that everything was in S/N units,
#	# which is inconvenient since the amplitude prior is in real amplitude.
#	# but we know our noise properties in terms of amplitude units from
#	# iA, arhs, so all of the above can be expressed in physical units instead.
#	def calc_log_prior(self, x):
#		r     = (1e-10+np.sum(x[:2]**2))**0.5
#		logp  = soft_prior(r, self.rmax)
#		logp += np.sum(soft_prior(-x[2:], 0))
#		return logp
#	def calc_dlog_prior(self, x):
#		r     = (1e-10+np.sum(x[:2]**2))**0.5
#		dlogp = np.zeros(len(x))
#		dlogp[:2] = x[:2]/r*soft_prior(r, self.rmax, deriv=True)
#		dlogp[2:] = -soft_prior(-x[2:], 0, deriv=True)
#		return dlogp
#	def calc_initial_value(self):
#		return np.zeros(2, self.dtype)
#	def calc_amp_eqsys(self, x):
#		"""Compute the left and right hand side of the conditional amplitude
#		distribution, returning iA, arhs.
#		iA   = P'N"P, arhs = P'N"d, ahat = iA"arhs
#		"""
#		P    = self.calc_P(x)
#		iNP  = self.get_cache("iNP", x, lambda: np.array([self.iN[i].dot(P[i]) for i in range(self.nmap)]))
#		PNP  = np.zeros([self.nmap, self.nmap], self.dtype)
#		PNB  = np.zeros([self.nmap, self.npix], self.dtype)
#		# Everything is block-diagonal, except core, which we have already handled
#		for i in range(self.nmap):
#			PNP[i,i] = P[i].dot(iNP[i])
#			PNB[i]   = self.B[i].dot(iNP[i]) # B and iN are symmetric
#		iA    = PNP - PNB.dot(self.icore.dot(PNB.T))
#		arhs  = np.sum(P*self.iNtotm,1)
#		return iA, arhs
#	def calc_damp_eqsys(self, x):
#		"""Compute what we need to get the derivative of ahat with respect to x.
#		da/dx = iA"*diA*iA" * arhs + iA" * darhs
#		darhs = dP' N" d
#		diA   = dP' N" P + P' N" dP"""
#		P    = self.calc_P(x)
#		iNP  = self.get_cache("iNP", x, lambda: np.array([self.iN[i].dot(P[i]) for i in range(self.nmap)]))
#		diA  = np.zeros([self.nx, self.nmap, self.nmap], self.dtype)
#		darhs= np.zeros([self.nx, self.nmap], self.dtype)
#		for c in range(self.nx):
#			dP   = self.calc_P(x, deriv=c)
#			dPNd = np.zeros([self.nmap, self.npix], self.dtype)
#			dPNP = np.zeros([self.nmap, self.nmap], self.dtype)
#			dPNB = np.zeros([self.nmap, self.npix], self.dtype)
#			PNB  = np.zeros([self.nmap, self.npix], self.dtype)
#			for i in range(self.nmap):
#				dPNP[i,i]  = dP[i].dot(iNP[i])
#				PNB[i]     = self.B[i].dot(iNP[i])
#				dPNB[i]    = self.B[i].dot(self.iN[i].dot(dP[i]))
#			diA[c]   = dPNP - dPNB.dot(self.icore.dot(PNB.T))
#			diA[c]  += diA[c].T
#			darhs[c] = dP.dot(self.iNtotm)
#		return diA, darhs
#	def calc_model(self, x):
#		amps = x[self.aoff:][self.groups]
#		P    = self.calc_P(x)
#		model= np.zeros([self.nmap, self.npix], self.dtype)
#		for i in range(self.nmap):
#			model[i] = amps[i]*P[i]
#		return model
#	def calc_dmodel(self, x):
#		"""Calculate the derivative of the model with respect to all the parameters.
#		This will result in nmap*nparam maps, but the maps are tiny, so this should be fine."""
#		pos, amp = x[:2], x[2:]
#		dmodel = np.zeros([self.nparam, self.nmap, self.npix], self.dtype)
#		# First get the derivative by position
#		for i in range(2):
#			dP = self.calc_P(x, pos_deriv=i)
#			for gi, g in enumerate(self.groups):
#				dmodel[i,gi] = amp[g]*dP[gi]
#		# Then get the derivative by amplitude
#		x_noamp     = x.copy()
#		x_noamp[2:] = 1.0
#		unit_model  = self.calc_model(x_noamp)
#		for gi, g in enumerate(self.groups):
#			dmodel[2+g,gi] = unit_model[gi]
#		return dmodel
#	def calc_dmodel_num(self, x, delta=1e-3):
#		res = np.zeros([len(x),self.nmap,self.npix],self.dtype)
#		for i in range(len(x)):
#			x1 = x.copy(); x1[i] -= delta
#			y1 = self.calc_model(x1)
#			x2 = x.copy(); x2[i] += delta
#			y2 = self.calc_model(x2)
#			res[i] = (y2-y1)/(2*delta)
#		return res
#	def calc_Q(self, x, deriv=None):
#		# Use periodic shifting to avoid pixel-border derivative problems
#		pos_mat = fft.shift(self.pos_base, x, deriv=deriv).reshape(self.npix)
#		return [pos_mat]*self.nmap
#	def calc_P(self, x, deriv=None):
#		"""Returns the matrix P that takes us from our full amplitudes
#		to the full model. While P is logically [nmap*npix,nmap], the amplitudes do not
#		mix, so it is enough to return the [nmap,npix] non-zero entries.
#		P_full[map1*pix,map2] = P_small[map2,pix]*delta(map1,map2).
#		"""
#		return self.get_cache("P",
#				np.concatenate([x,[-1 if deriv is None else deriv]]),
#				lambda: np.array([self.B[i].dot(Q) for i,Q in enumerate(
#					self.calc_Q(x, deriv=deriv))]))
#	def mul_iNtot(self, m):
#		"""Ntot" = N" - N"Bb(S" + b'B'N"Bb)"b'B'N" = N" - N"Bb icore b'BN"""
#		iNm   = np.zeros([self.nmap, self.npix], self.dtype)
#		cbBiN = np.zeros(self.npix, self.dtype)
#		for i in range(self.nmap):
#			iNm[i] = self.iN[i].dot(m[i])
#			cbBiN  += self.B[i].dot(iNm[i])
#		cbBiN = self.icore.dot(cbBiN)
#		for i in range(self.nmap):
#			iNm[i] -= self.iN[i].dot(self.B[i].dot(cbBiN))
#		return iNm
#	def get_cache(self, key, x, f):
#		if key not in self.cache or not np.allclose(x, self.cache[key][0], rtol=1e-14, atol=0):
#			self.cache[key] = [np.array(x), np.array(f())]
#		return self.cache[key][1].copy()
#	def format_sample(self, x, amps):
#		return "%8.3f %8.3f" % tuple(x) + " %8.3f"*(len(amps))%tuple(amps/1e3)
#	def maximize(self, x0=None, verbose=False):
#		"""Find the maximum likelihood point, along with a fisher error estimate."""
#		if x0 is None:
#			x0 = self.calc_initial_value()
#		self.n  = 0
#		def f(x):
#			self.n += 1
#			t1 = time.time()
#			p, info = self.calc_log_posterior(x, return_info=True)
#			t2 = time.time()
#			if verbose:
#				print "%3d %5.2f %9.3f %s" % (self.n, t2-t1, p, self.format_sample(x, info.ahat))
#				sys.stdout.flush()
#			return p
#		x, logP, dlogP, ihess, nf, ng, w = optimize.fmin_bfgs(f, x0, self.calc_dlog_posterior, disp=False, full_output=True)
#		# The returned inverse hessian is not very reliable. Use
#		# numerical derivative instead
#		ddlogP = self.calc_ddlog_posterior_num(x)
#		x_cov  = np.linalg.inv(ddlogP)
#
#		# Get the full amps too
#		iA, arhs = self.calc_amp_eqsys(x)
#		A = utils.eigpow(iA, -1)
#		a = A.dot(arhs)
#		# Estimate the equivalent S/N
#		xtmp = x.copy(); xtmp[self.aoff:] = 0
#		logP_null = self.calc_log_posterior(xtmp)
#		sn = max(2*(logP_null-logP)-self.nparam,0)**0.5
#		# Get the gaussian errors
#		return bunch.Bunch(x=x, x_cov=x_cov, sn=sn, logP=logP, logP_null=logP, afull=a, afull_cov=A)







def find_candidates(snmap, lim=5.0, edge=0):
	"""Given a S/N ratio map, find sources with S/N above lim while avoiding the
	given amount of pixels around the edge. Returns a record array sorted by S/N,
	with the fields [sn, type, pos[2], pix[2], npix[2]]."""
	dtype  = [("sn","f"),("type","S10"),("pos","2f"),("pix","2f"),("npix","i")]
	ny,nx= snmap.shape
	mask = enmap.zeros(snmap.shape, snmap.wcs, np.bool)
	mask[edge:ny-edge,edge:nx-edge] = (snmap > lim)[edge:ny-edge,edge:nx-edge]
	labels, nlabel = ndimage.label(mask)
	res  = np.recarray(nlabel, dtype)
	if len(res) > 0:
		res.pix  = np.array(ndimage.center_of_mass(snmap**2, labels, np.arange(1,nlabel+1)))
		res.pos  = snmap.pix2sky(res.pix.T).T
		res.sn   = ndimage.maximum(snmap, labels, np.arange(1,nlabel+1))
		res.npix = ndimage.sum(snmap*0+1, labels, np.arange(1,nlabel+1))
		inds = np.argsort(res.sn[::-1])
		res = res[inds]
	return res

def prune_candidates(cands, others=None, scale=2.0, xmax=7, tol=0.1, verbose=False, other_scale=0.05,
		other_dist=1*utils.arcmin):
	"""Prune false positives caused by ringing in the filter. We assume that the
	ringing goes as 1/(1+x)**3 where x = r/scale, up to xmax beyond which it is
	zero. This is tuned to ACT beam etc.. What I really should use is the
	filter profile here, but this will do for now."""
	inds  = np.argsort(cands.sn)[::-1]
	cands = cands[inds]
	def calc_leak(x, sn_ratio): return sn_ratio/(1+x)**3*(x < xmax)
	# To avoid double-disqualification, we will loop from strongest to weakest
	ocands = cands[:1]
	if others is None:
		nother = 0
		odists = None
	else:
		# This is to support avoiding strong candidates found in any previous passes.
		# Prepend down-scaled versions of the previous candidates because we assume that
		# subtraction is not perfect.
		dummy     = np.recarray(len(others), cands.dtype)
		dummy.sn  = others.sn*other_scale
		dummy.pos = others.pos
		ocands    = np.rec.array(np.concatenate([dummy,ocands]))
		nother    = len(others)
		odists    = np.min(calc_dist(cands.pos.T[:,:,None], others.pos.T[:,None,:]),1)
	for i in range(1, len(cands)):
		if odists is not None and odists[i] < other_dist: continue
		dists = calc_dist(ocands.pos.T,cands.pos.T[:,i,None])/utils.arcmin
		leaks = calc_leak(dists/scale, ocands.sn/cands.sn[i])
		if np.max(leaks) < tol:
			ocands = np.rec.array(np.concatenate([ocands, cands[i:i+1]]))
		if verbose:
			j = np.argmax(leaks)
			cand, ocan = cands[i], ocands[j]
			print "%3d %5s %8.3f %8.3f %8.3f  leak %8.3f %3d %5s %8.3f %8.3f %8.3f" % (
					i, cand.type, cand.sn, cand.pos[0]/utils.degree, cand.pos[1]/utils.degree, leaks[j],
					j, ocan.type, ocan.sn, ocan.pos[0]/utils.degree, ocan.pos[1]/utils.degree)
	# Get rid of any prepended candidates
	ocands = ocands[nother:]
	return ocands

def format_catalogue(cat):
	"""Format an object catalogue for screen display. This prints the most important entries"""
	if cat.ndim == 0: cat = np.rec.array(cat[None])
	res = ""
	for i, c in enumerate(cat):
		res += "%5s %7.2f %7.2f" % (c.type, c.sn, c.sn0)
		res += "  %8.3f %5.3f %8.3f %5.3f" % (c.pos[0], c.dpos[0], c.pos[1], c.dpos[1])
		res += "  %5.2f %5.2f  %10.3f %7.3f" % (c.fwhm, c.dfwhm, c.amp, c.damp)
		res += "\n"
	return res

def div_nonan(a,b,fill=0):
	with utils.nowarn(): res = a/b
	res[~np.isfinite(res)] = fill
	return res

def ps2d_to_mat(ps2d, n):
	corrfun = map_ifft(ps2d+0j)/(ps2d.shape[-2]*ps2d.shape[-1])**0.5
	thumb   = corrfun_thumb(corrfun, n)
	mat     = corr_to_mat(thumb, n)
	return mat

def extract_thumb_roll(map, pix, n):
	return np.roll(np.roll(map, n//2-pix[1], -1)[...,:n], n//2-pix[0], -2)[...,:n,:]

def extract_thumb(map, pix, n):
	y1, x1 = pix[0]-n//2, pix[1]-n//2
	y2, x2 = y1+n, x1+n
	if y1 >= 0 and y2 < map.shape[-2] and x1 >= 0 and x2 < map.shape[-1]:
		return map[...,y1:y2,x1:x2].copy()
	else:
		return extract_thumb_roll(map, pix, n)

def expand_thumb(thumb, pix, shape):
	res = np.zeros(shape)
	res[...,:thumb.shape[-2],:thumb.shape[-1]] = thumb
	return np.roll(np.roll(res, -(thumb.shape[-1]//2-pix[-1]), -1), -(thumb.shape[-2]//2-pix[-2]), -2)

def corrfun_thumb(corr, n):
	tmp = np.roll(np.roll(corr, n, -1)[...,:2*n], n, -2)[...,:2*n,:]
	return np.roll(np.roll(tmp, -n, -1), -n, -2)

def corr_to_mat(corr, n):
	res = enmap.zeros([n,n,n,n],dtype=corr.dtype)
	for i in range(n):
		tmp = np.roll(corr, i, 0)[:n,:]
		for j in range(n):
			res[i,j] = np.roll(tmp, j, 1)[:,:n]
	return res

def shift_nonperiodic(a, shift, axes=None, pad=100, deriv=None):
	apad = np.pad(a, pad, "constant")
	opad = fft.shift(apad, shift, axes=axes, deriv=deriv)
	return opad[(slice(pad,-pad),)*apad.ndim]

def project_out(imat, modes, frac=1.0):
	imat  = np.asarray(imat)
	modes = np.asarray(modes)
	if modes.ndim == 1: modes = modes[:,None]
	core = modes.T.dot(imat).dot(modes)
	return imat - frac*imat.dot(modes).dot(np.linalg.solve(core, modes.T.dot(imat)))

def binvec(vec, inds):
	return np.bincount(inds, vec)
def binmat(mat, inds):
	inds = np.asarray(inds)
	nin  = mat.shape[0]
	nout = np.max(inds)+1
	inds_2d = inds[np.mgrid[:nin,:nin]].reshape(2,-1)
	bins = np.ravel_multi_index(inds_2d, (nout, nout))
	omat = np.bincount(bins, mat.reshape(-1)).reshape(nout,nout)
	return omat

def draw_emcee_stretch(scale):
	# scale pdf: 1/sqrt(z) between 1/a and a. so cdf is:
	# int_(1/a)^x z**-0.5 da = 2[z**0.5]_(1/a)^x = 2*(x**0.5 - a**-0.5)
	# normalize: (a**0.5-a**-0.5)**-1 * (x**0.5-a**-0.5)
	# inverse: ((a**0.5-a**-0.5)*p+a**-0.5)**2
	a = scale
	return (np.random.uniform(0,1)*(a**0.5-a**-0.5)+a**-0.5)**2

def log_prob_gauss_positive(mu, cov):
	"""Returns the probability that all gaussian variables with mean mu and cov cov
	will be positive."""
	mu = np.asarray(mu)
	zero = np.zeros(mu.shape)
	# logcdf is not as accurate as the log promises
	try:
		return stats.multivariate_normal.logcdf(zero, -mu, cov)
	except np.linalg.LinAlgError:
		# If we failed, try ignoring the correlations
		x = mu/np.diag(cov)**0.5
		if np.all(np.isfinite(x)):
			return np.sum([log_prob_gauss_positive_single(val) for val in x])
		else: return -np.inf

def log_prob_gauss_positive_single(x, nmax=10):
	if x > -10: return np.log(0.5*special.erfc(-x/2**0.5))
	pre  = -0.5*np.log(2*np.pi) - np.log(-x) - 0.5*x**2
	rest = 1.
	fact = 1.
	div  = 1.
	for n in xrange(1, nmax):
		fact *= -(2*n-1)
		div  *= x**2
		rest += fact/div
	res = pre + np.log(rest)
	return res

# we want an accurate log erf.
# erf(x) = 2/sqrt(pi) int^x exp(-t**2) dt
#        = 2/sqrt(pi) exp(-x**2) int^x exp(-(t**2-x**2)) dt
# y = t-x: t**2-x**2 = y**2 + 2xy,
#        = 2/sqrt(pi) exp(-x**2) [int^0 exp(-y**2) exp(-2xy) dy]

#def log_prob_gauss_positive_dmu(mu, cov):
#	# d logcdf = 1/cdf * dcdf
#	mu   = np.asarray(mu)
#	if mu.ndim > 0:
#		E, V = np.linalg.eigh(cov)
#		a    = V.T.dot(mu)
#	else:
#		a, E = mu, cov
#	# dcdf = d/da int^a dx (2pi)**-0.5 * exp(-0.5*x**2) = (2pi)**-0.5 * exp(-0.5*a**2)
#	log_dcdf = -0.5*mu.size*np.log(2*np.pi) -0.5*(a/E)**2
#	log_cdf  = log_prob_gauss_positive(mu, cov)
#	log_dlogcdf = log_dcdf - log_cdf
#	return np.exp(log_dlogcdf)
#
#def log_prob_gauss_positive_dcov_numerical(mu, cov, step=1e-8):
#	# Perform derivative numerically, since the analytical case looks tricky.
#	# But don't directly differentiate its elements, since that can be unstable.
#	# instead write Y = Eh" V'CV Eh", which will be I for C = cov and very close to
#	# I for the displaced versions. We want to perform the deriative with respect to
#	# Y, and then translate the result back.
#
#	# If d(f(AXB)) = A f'(AXB) B.
#
#	# d f(X) = d f(UYU') = U f'(UYU') U'
#
#	E, V = np.linalg.eigh(cov)
#	U  = V*E[None,:]**0.5
#	iU = V*E[None,:]**-0.5
#	def f(Y): return log_prob_gauss_positive(mu, U.dot(Y).dot(U.T))
#	return iU.T.dot(numerical_derivative(f, np.eye(len(mu)), step=step)).dot(iU)

def numerical_derivative(f, x, step=1e-8):
	x   = np.array(x, dtype=float)
	xf  = x.reshape(-1)
	res = []
	for i in range(len(xf)):
		x1 = xf.copy(); x1[i] += step; x1 = x1.reshape(x.shape)
		x2 = xf.copy(); x2[i] -= step; x2 = x2.reshape(x.shape)
		deriv = (f(x1)-f(x2))/(2*step)
		res.append(deriv)
	res = np.array(res)
	return res.reshape(x.shape + res.shape[1:])

# But we also need the derivative of the gauss positive with respect to the covariance.
# This is really nasty.

def calc_dist(decra1, decra2):
	diff     = decra1 - decra2
	diff[1] *= np.cos(0.5*(decra1[0]+decra2[0]))
	dist     = np.sum(diff**2,0)**0.5
	return dist

def soft_prior(v, vmax, dv=0.01, deriv=False):
	with utils.nowarn():
		res = np.exp((v-vmax)/dv)
		if deriv: return res/dv
		else: return res

def pbox_out_of_bounds(pbox, shape, wcs):
	if pbox[0,0] >= shape[-2] or pbox[1,0] <= 0: return True
	nx = np.abs(utils.nint(360./wcs.wcs.cdelt[0]))
	xr = pbox[:,1]-pbox[0,1]//nx*nx
	xw = xr+nx
	if (xr[1] <= 0 or xr[0] >= shape[-1]) and (xw[1] <= 0 or xw[0] >= shape[-1]): return True
	return False

def apod_mask_edge(mask, n):
	dist = ndimage.distance_transform_edt(mask)/n
	x    = np.minimum(1,dist)
	return 0.5*(1-np.cos(np.pi*x))

def write_catalogue(fname, cat, box=None):
	hdu = fits.hdu.table.BinTableHDU(cat)
	if box is not None:
		for key, val in zip(["DEC1","RA1","DEC2","RA2"], box.reshape(-1)/utils.degree):
			hdu.header.append((key, "%12.8f" % val))
	hdu.writeto(fname, overwrite=True)

def write_catalogue_table(fname, cat):
	"""Write cataloge to fits using the astropy table interface. Does not support
	storing the bounding box."""
	table.Table(info.catalogue).write(fname, overwrite=True)

def read_catalogue(fname, return_box=False):
	hdu = fits.open(fname)[1]
	cat = np.rec.array(np.asarray(hdu.data))
	if return_box:
		box = []
		for key in ["DEC1","RA1","DEC2","RA2"]:
			box.append(float(hdu.header[key])*utils.degree)
		box = np.array(box).reshape(2,2)
		return cat, box
	else: return cat

def estimate_separable_pixwin_from_normalized_ps(ps2d):
	mask = ps2d < 2
	res  = []
	for i in range(2):
		profile  = np.sum(ps2d*mask,1-i)/np.sum(mask,1-i)
		profile /= np.percentile(profile,90)
		profile  = np.fft.fftshift(profile)
		edge     = np.where(profile >= 1)[0]
		if len(edge) == 0:
			res.append(np.full(len(profile),1.0))
			continue
		edge = edge[[0,-1]]
		profile[edge[0]:edge[1]] = 1
		profile  = np.fft.ifftshift(profile)
		# Pixel window is in signal, not power
		profile **= 0.5
		res.append(profile)
	return res
