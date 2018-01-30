import numpy as np, os, time, imp
from scipy import ndimage, optimize
from enlib import enmap, retile, utils, bunch, cg, fft, powspec

def read_config(fname):
	config = imp.load_source("config", fname)
	config.path = fname
	return config

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
		dataset.beam  = setup_beam(dataset.beam_params, workdir=cdir)
		dataset.box   = enmap.box(shape, wcs, corner=False)
		dataset.config= config # pretty backwards...
	return datasets

def read_geometry(fname):
	if os.path.isdir(fname):
		geo = retile.read_tileset_geometry(fname + "/tile%(y)03d_%(x)03d.fits")
		return geo.shape, geo.wcs
	else:
		return enmap.read_map_geometry(fname)

def setup_beam(params, nl=50000, workdir="."):
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
		# Fit gaussian that would reach res[ndata-1] by that point. We do this
		# because we want well-defined values for every l, to allow us to divide later
		# -0.5*(ndata-1)**2*sigma**2 = res[ndata-1] => sigma**2 = -2*res[ndata-1]/(ndata-1)**2
		sigma2 = -2*res[ndata-1]/(ndata-1)**2
		res[ndata:] = -0.5*l[ndata:]**2*sigma2
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

def make_dummy_tile(shape, wcs, box, pad=0):
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
				apod = apod.apod(apod_edge)
				split.data.div *= apod
				split.data.H   = split.data.div**0.5
			dataset.ref_div = np.sum([split.ref_div for split in dataset.splits])
		tot_ref_div = np.sum([dataset.ref_div for dataset in datasets])

		ly, lx   = enmap.laxes(self.shape, self.wcs)
		lr       = (ly[:,None]**2 + lx[None,:]**2)**0.5
		bmin = np.min([beam_size(dataset.beam) for dataset in datasets])
		# If no target beam was specified, set it to be the best of the input beams
		if ref_beam is None:
			ref_beam = datasets[0].beam
			for dataset in datasets[1:]:
				ref_beam = np.maximum(ref_beam, dataset.beam)
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
