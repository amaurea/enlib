import numpy as np, os, time, imp, copy, functools
from scipy import ndimage, optimize, interpolate
from enlib import enmap, retile, utils, bunch, cg, fft, powspec, array_ops
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
	return x*ex/(ex-1)*(x*(ex+1)/(ex-1)-4)

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

def sz_rad_projected(r_ort, fwhm=1.0, xmax=5, dx=1e-3):
	"""Compute the projected sz radial profile on the sky, as a function of
	the angular distance r_ort from the center, in arcminutes. The cluster profile full-width
	half-max is given by fwhm."""
	x_ort = np.asarray(r_ort)*0.34208/fwhm
	x_par = np.arange(0,xmax,dx)+dx/2
	res   = np.array([np.sum(sz_rad_core((x_par**2 + xo**2)**0.5)) for xo in x_ort.reshape(-1)])
	norm  = np.sum(sz_rad_core(x_par))
	return res.reshape(x_ort.shape)/norm

def sz_rad_projected_map(shape, wcs, fwhm=1.0, xmax=5, dx=1e-3):
	pos = enmap.posmap(shape, wcs)
	iy,ix = np.array(shape[-2:])//2+1
	r   = np.sum((pos-pos[:,iy,ix][:,None,None])**2,0)**0.5
	r   = np.roll(np.roll(r,-ix,-1),-iy,-2)
	# Next build a 1d spline of the sz cluster profile covering all our radii
	r1d = np.arange(0, np.max(r)*1.1, min(r[0,1],r[1,0])*0.5)
	y1d = sz_rad_projected(r1d/utils.arcmin, fwhm)
	spline = interpolate.splrep(r1d, np.log(y1d))
	y2d = enmap.ndmap(np.exp(interpolate.splev(r, spline)), wcs)
	return y2d

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
	if len(map) == ncomp: return map
	omap = enmap.zeros((ncomp,)+map.shape[-2:], map.wcs, map.dtype)
	omap[:len(map)] = map
	omap[len(map):] = np.random.standard_normal((len(map),)+map.shape[-2:])*np.std(map)*rms_factor
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
				apod*= apod.apod(apod_edge)
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
	def read(self, box, pad=0, prune=True, verbose=False, cache_dir=None, dtype=None, div_unhit=1e-7, read_cache=False, ncomp=None):
		"""Read the data for each of our datasets that falls within the given box, returning a new Mapset
		with the mapset.datasets[:].split[:].data member filled with a map and div. If prune is False,
		then the returned mapset will have the same maps as the original mapset. If prune is True (the default),
		on the other hand, splits with no data are removed, as are datasets with too few splits. This can
		result in all the data being removed, in which case None is returned."""
		res = self.copy()
		for dataset in res.datasets:
			pbox = calc_pbox(dataset.shape, dataset.wcs, box)
			pbox[0] -= pad
			pbox[1] += pad
			psize = pbox[1]-pbox[0]
			ffpad = np.array([fft.fft_len(s, direction="above")-s for s in psize])
			pbox[1] += ffpad

			dataset.pbox  = pbox
			dataset.ngood = 0
			for split in dataset.splits:
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
		if len(res.datasets) == 0: return None
		# Extra information about what we read
		res.ffpad    = ffpad
		res.shape, res.wcs = common_geometry([split.data.map.geometry for dataset in res.datasets for split in dataset.splits if split.data is not None], ncomp=ncomp)
		res.dtype    = res.datasets[0].splits[0].data.map.dtype
		res.l        = enmap.modlmap(res.shape, res.wcs)
		return res

def sanitize_maps(mapset, map_max=1e8, div_tol=20, apod_val=0.2, apod_alpha=5, apod_edge=60):
	"""Get rid of extreme values in maps and divs, and further downweights the the edges and
	faint regions of div."""
	ncomp = max([split.data.map.preflat.shape[0] for dataset in mapset.datasets for split in dataset.splits if split.data is not None])
	for dataset in mapset.datasets:
		for split in dataset.splits:
			if split.data is None: continue
			split.ref_div = robust_ref(split.data.div)
			# Avoid single, crazy pixels
			split.data.div = np.minimum(split.data.div, split.ref_div*div_tol)
			split.data.div = filter_div(split.data.div)
			split.data.map = np.maximum(-map_max, np.minimum(map_max, split.data.map))
			# Expand map to ncomp components
			split.data.map = add_missing_comps(split.data.map, ncomp)
			# Build apodization
			split.data.apod  = np.minimum(split.data.div/(split.ref_div*apod_val), 1.0)**apod_alpha
			split.data.apod *= split.data.apod.apod(apod_edge)
			# And apply it
			split.data.div *= split.data.apod
		dataset.ref_div = np.sum([split.ref_div for split in dataset.splits if split.data is not None])
	mapset.ref_div = np.sum([dataset.ref_div for dataset in mapset.datasets])

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
		# Then use it to build the diff maps and noise spectra
		dset_ps = None
		for split in dataset.splits:
			if split.data is None: continue
			diff  = split.data.map - dset_map
			# We can't whiten diff with just H.
			# diff = m_i - sum(div)"sum(div_j m_j), and so has
			# var  = div_i" + sum(div)"**2 * sum(div) - 2*sum(div)"div_i/div_i
			#      = div_i" - sum(div)"
			# ivar = (div_i" - sum(div)")"
			with utils.nowarn():
				diff_H   = (1/split.data.div - 1/dset_div)**-0.5
			diff_H[~np.isfinite(diff_H)] = 0
			wdiff = diff * diff_H
			# What is the healthy area of wdiff? Wdiff should have variance
			# 1 or above. This tells us how to upweight the power spectrum
			# to take into account missing regions of the diff map.
			ndown = 10
			wvar  = enmap.downgrade(wdiff**2,ndown)
			goodfrac_var  = np.sum(wvar > 1e-3)/float(wvar.size)
			goodfrac_apod = np.mean(split.data.apod)
			goodfrac = min(goodfrac_var, goodfrac_apod)
			if goodfrac < 0.1: goodfrac = 0
			ps    = np.abs(map_fft(wdiff))**2
			# correct for unhit areas, which can't be whitend
			with utils.nowarn(): ps   /= goodfrac
			if dset_ps is None:
				dset_ps = enmap.zeros(ps.shape, ps.wcs, ps.dtype)
			dset_ps += ps
			nsplit += 1
		if nsplit < 2: continue
		dset_ps /= nsplit
		# Smooth ps to reduce sample variance
		dset_ps  = smooth_ps(dset_ps, ps_res, ndof=2*(nsplit-1))
		# For planck, the upscaling to act resolution results in no signal nor noise
		# at high l. This is creating numerical problems. I should get to the bottom
		# of this. For now, this mostly avoids them, and should be safe, as typical
		# values of dset_ps are 1, and so 1e-3 should not occur normally.
		dset_ps  = np.maximum(dset_ps, 1e-3)
		# If we have invalid values, then this whole dataset should be skipped
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
	bmin    = np.min([beam_size(dataset.beam) for dataset in mapset.datasets])
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

def setup_profiles_ptsrc(mapset):
	setup_profiles_helper(mapset, lambda freq: calc_profile_ptsrc(freq, nl=mapset.nl))
def setup_profiles_sz(mapset, scale):
	# sz must be evaluated in real space to avoid aliasing. First get the
	# distance from the corner, including wrapping. We do this by getting the
	# distance from the center, and then moving the center to the corner afterwards
	y2d = sz_rad_projected_map(mapset.shape, mapset.wcs, fwhm=scale)
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
		param = (d.beam_params, d.window_params)
		if param not in cache:
			beam_2d = eval_beam(d.beam, mapset.l)
			# Apply pixel window
			if d.window_params[0] == "native":
				wy, wx = enmap.calc_window(beam_2d.shape)
				beam_2d *= wy[:,None]
				beam_2d *= wx[None,:]
			elif d.window_params[0] == "none":
				pass
			elif d.window_params[0] == "lmax":
				beam_2d[mapset.l>d.window_params[1]] = 0
			else: raise ValueError("Unrecognized pixel window type '%s'" % (d.window_params[0]))
			cache[param] = beam_2d
		d.beam_2d = cache[param]

def setup_background_cmb(mapset, cl):
	# The background is shared between all datasets
	cl_TT = cl.reshape(-1,cl.shape[-1])[0]
	mapset.S = enmap.spec2flat(mapset.shape, mapset.wcs, cl_TT[None,None])[0,0]

class SignalFilter:
	def __init__(self, mapset):
		self.mapset = mapset
		# Extract and flatten all our input maps
		self.m  = [split.data.map.preflat[0]  for dataset in mapset.datasets for split in dataset.splits]
		self.H  = [split.data.H               for dataset in mapset.datasets for split in dataset.splits]
		self.iN = [dataset.iN.preflat[0]      for dataset in mapset.datasets for split in dataset.splits]
		self.hN = [dataset.iN.preflat[0]**0.5 for dataset in mapset.datasets for split in dataset.splits]
		self.Q  = [dataset.signal_profile_2d  for dataset in mapset.datasets for split in dataset.splits]
		self.B  = [dataset.beam_2d            for dataset in mapset.datasets for split in dataset.splits]
		#print "FIXME"
		#self.m, self.H, self.iN, self.hN, self.Q, self.B = [w[:1] for w in [self.m, self.H, self.iN, self.hN, self.Q, self.B]]
		self.shape, self.wcs = self.m[0].geometry
		self.dtype= mapset.dtype
		self.ctype= np.result_type(self.dtype,0j)
		self.npix = self.shape[0]*self.shape[1]
		self.nmap = len(self.m)
		self.tot_div = enmap.zeros(self.shape, self.wcs, self.dtype)
		for H in self.H: self.tot_div += H**2
	def calc_rhs(self):
		# Calc rhs = Ch H m. This is effectively a set of fully whitened maps.
		# rhs is returend in fourier space
		rhs = [self.hN[i]*map_fft(self.H[i]*self.m[i]) for i in range(self.nmap)]
		return rhs
	def calc_mu(self, rhs, maxiter=250, cg_tol=1e-7, verbose=False, dump_dir=None):
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
		# system.
		Hmean  = [np.mean(H) for H in self.H]
		V = enmap.zeros((self.nmap,)+self.shape, self.wcs, self.dtype)
		for i in range(self.nmap):
			V[i] = self.mapset.S**0.5*self.B[i]*Hmean[i]*self.hN[i]
		iprec  = np.eye(self.nmap)[:,:,None,None] + V[:,None]*V[None,:]
		prec   = array_ops.eigpow(iprec, -1, (0,1), lim=1e-8)
		#prec_diag = [1/(1+self.iN[i]*Hmean[i]**2*self.B[i]**2*self.mapset.S) for i in range(self.nmap)]
		def M(x):
			fmaps = unzip(x)
			omaps = enmap.map_mul(prec, fmaps)
			#omaps  = fmaps*prec_diag
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
		alpha = enmap.zeros(self.shape, self.wcs, self.dtype)
		for i in range(self.nmap):
			alpha += map_ifft(self.Q[i]*self.B[i]*map_fft(self.H[i]*map_ifft(self.hN[i]*mu[i])))
		return alpha
	def calc_alpha_simple(self):
		# Plain matched filter for a single map
		Hmean = np.median(self.H[0])
		Ntot  = self.mapset.S*self.B[0]**2 + 1/np.maximum(Hmean**2*self.iN[0],1e-25)
		filter= self.Q[0]*self.B[0]/Ntot
		return map_ifft(filter*map_fft(self.m[0])), Ntot
	def calc_dalpha_analytical(self):
		# Estimate the rms of the alpha map. There's something weird about
		# the estimates I'm getting from this method currently - they
		# have a variance about 7 times too high.
		# The covariance is given by A = P'(BSB'+N)"P, which is hard to evaluate.
		# We expand it as A = P'N"P - P'N"B(S"+B'N"B)"B'N"P, and then use
		# a harmonic-only approximation for most of the second term.
		Hmean = [np.mean(H) for H in self.H]
		# 1. Compute the woodbury core. This is common for all maps
		BiNB = self.iN[0]*0
		for i in range(self.nmap):
			BiNB += self.B[i]**2*Hmean[i]**2*self.iN[i]
		with utils.nowarn():
			wood = 1/(1/self.mapset.S + BiNB)
		# 2. Loop through maps, computing their contribution to the variance
		alpha_var = 0
		for i in range(self.nmap):
			harm2 = self.Q[i]**2*self.B[i]**2*self.iN[i]**2*Hmean[i]**2*self.B[i]**2*wood
			harm1 = self.Q[i]**2*self.B[i]**2*self.iN[i]
			alpha_var += self.H[i]**2 * spec2var(harm1-harm2)
		alpha_rms = alpha_var**0.5
		return alpha_rms
	def calc_dalpha_empirical(self, alpha, bsize=50):
		# Estimate the rms of the alpha map based on its actual variance.
		# We assume that the per pixel variance will be proprotional to tot_div,
		# with an unknown overall scale that we fit from the alpha map itself
		avars = enmap.downgrade(alpha**2, bsize)
		dvals = enmap.downgrade(self.tot_div, bsize)
		mask  = np.isfinite(avars)&np.isfinite(dvals)
		avars, dvals = avars[mask], dvals[mask]
		avars, dvals = avars[dvals>0], dvals[dvals>0]
		ratios= avars/dvals
		mask  = (ratios > np.median(ratios)/2)&(ratios < np.median(ratios)*2)
		avars, dvals = avars[mask], dvals[mask]
		scale = np.sum(avars*dvals)/np.sum(dvals**2)
		alpha_rms = (self.tot_div * scale)**0.5
		return alpha_rms
	def calc_alpha_cov_harmonic(self):
		# The harmonic-only approximation to alpha's covariance A" = P'N"P is
		# given by A" = P'N"P = (P'[BSB'+ H"C"H"]"P) sim (P'HCh[ChHBSB'HCh+1]"ChHP)
		# V = ChHBSh => P'HCh[VV'+1]"ChHP,
		# P'HCHP - P'HChV(1+V'V)"V'ChHP
		# P'HCHP - P'HCHBSh(1+ShB'HCHBSh)"Sh'BHCHP
		# Everything is diagonal in fourier space here. The result will be the
		# fourier-space representation of the covariance, but it will be real
		# because it's a 2d power spectrum
		Hmean = [np.mean(H) for H in self.H]
		PHCHP = enmap.zeros(self.shape, self.wcs, self.dtype)
		PHCHB = enmap.zeros(self.shape, self.wcs, self.dtype)
		core  = enmap.zeros(self.shape, self.wcs, self.dtype)
		for i in range(self.nmap):
			PHCHP += self.Q[i]**2*self.B[i]**2*Hmean[i]**2*self.iN[i]
			PHCHB += self.Q[i]*self.B[i]**2*Hmean[i]**2*self.iN[i]
			core  += Hmean[i]**2*self.iN[i]*self.B[i]**2
		enmap.write_map("test_phchp.fits", np.fft.fftshift(PHCHP))
		core  = 1+self.mapset.S*core
		term2 = PHCHB**2*self.mapset.S/core
		iA = PHCHP - term2
		enmap.write_map("test_iA.fits", np.fft.fftshift(iA))
		return iA
	def calc_filter_harmonic(self):
		# The harmonic-only representation of the total filter
		# F such that alpha = Fm, e.g. F = P'N" = P'(BSB'+H"C"H")"
		# sim P'HCH - P'HCHBSh(1+ShB'HCHBSh)"Sh'BHCH
		# So this is the cov harmonic but with one less beam and without
		# the inversion at the end.
		Hmean = [np.mean(H) for H in self.H]
		sumPHCH = enmap.zeros(self.shape, self.wcs, self.dtype)
		PHCHB   = enmap.zeros(self.shape, self.wcs, self.dtype)
		sumHCHB = enmap.zeros(self.shape, self.wcs, self.dtype)
		core  = enmap.zeros(self.shape, self.wcs, self.dtype)
		for i in range(self.nmap):
			sumPHCH += self.Q[i]*self.B[i]*Hmean[i]**2*self.iN[i]
			PHCHB   += self.Q[i]*self.B[i]**2*Hmean[i]**2*self.iN[i]
			sumHCHB += self.B[i]*Hmean[i]**2*self.iN[i]
			core  += Hmean[i]**2*self.iN[i]*self.B[i]**2
		core  = 1+self.mapset.S*core
		term2 = PHCHB*sumHCHB*self.mapset.S/core
		res = sumPHCH - term2
		return res
	def sim(self):
		# Debug: make noise and signal simulations for each of our maps
		s = map_ifft(self.mapset.S**0.5*map_fft(enmap.rand_gauss(self.shape, self.wcs))).astype(self.dtype)
		n = enmap.samewcs([self.H[i]**-1*map_ifft(self.hN[i]**-1*map_fft(enmap.rand_gauss(self.shape, self.wcs))).astype(self.dtype) for i in range(self.nmap)],s)
		n += s
		return n
	def sim2(self):
		# Debug: test fourier space units by building a pure fourier-space sim. This ignores cmb correlations
		res = []
		for i in range(self.nmap):
			Hmean= np.mean(self.H[i])
			Ntot = self.mapset.S + 1/(np.maximum(self.iN[0],1e-10)*Hmean**2)
			n    = map_ifft(Ntot**0.5*map_fft(enmap.rand_gauss(self.shape, self.wcs).astype(self.dtype)))
			res.append(n)
		return res

def spec2var(spec_2d):
	return np.mean(spec_2d)**0.5

def blockvar(m, bsize=10):
	return enmap.downgrade(m**2, bsize)

