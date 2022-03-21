# Functions used by tenki/point_sources/sauron.py which I haven't found a better
# home for, and which might need some reworking to be generally usable.
import numpy as np, time, os
from numpy.lib import recfunctions
from pixell import enmap, utils, bunch, analysis, uharm, powspec, pointsrcs, curvedsky, mpi
from enlib import mapdata, array_ops, wavelets, multimap
from scipy import ndimage

def smooth_ps_angular(ps2d, brel=5):
	ps1d, l1d = ps2d.lbin(brel=brel)
	#np.savetxt("ps1d.txt", np.concatenate([l1d[None], ps1d.reshape(-1,len(l1d))],0).T, fmt="%15.7e")
	l = ps2d.modlmap()
	return enmap.samewcs(utils.interp(l, l1d, ps1d),ps2d)

def smooth_ps_gauss(ps2d, lsigma=200):
	"""Smooth a 2d power spectrum to the target resolution in l. Simple
	gaussian smoothing avoids ringing."""
	# This hasn't been tested in isolation, but breaks when used in smooth_ps_mixed
	# First get our pixel size in l
	ly, lx = enmap.laxes(ps2d.shape, ps2d.wcs)
	ires   = np.array([ly[1],lx[1]])
	sigma_pix = np.abs(lsigma/ires)
	fmap  = enmap.fft(ps2d)
	ky    = np.fft.fftfreq(ps2d.shape[-2])*sigma_pix[0]
	kx    = np.fft.fftfreq(ps2d.shape[-1])*sigma_pix[1]
	kr2   = ky[:,None]**2+kx[None,:]**2
	fmap *= np.exp(-0.5*kr2)
	return enmap.ifft(fmap).real

def smooth_downup(map, n):
	n = np.minimum(n, map.shape[-2:])
	o = (np.array(map.shape[-2:]) % n)//2
	return enmap.upgrade(enmap.downgrade(map, n, off=o, inclusive=True), n, off=o, oshape=map.shape, inclusive=True)

def smooth_ps_mixed(ps2d, brel=5, lsigma=100):
	# FIXME this doens't work. The noise matrix breaks when
	# inverted after doing this
	radial = smooth_ps_angular(ps2d, brel=brel)
	resid  = ps2d/radial
	resid  = smooth_ps_gauss(resid, lsigma=lsigma)
	model  = resid*radial
	return model

def dump_ps1d(fname, ps2d):
	ps, l = ps2d.lbin(brel=2)
	ps = powspec.sym_compress(ps)
	np.savetxt(fname, np.concatenate([l[None], ps],0).T, fmt="%15.7e")

# I don't like that these functions take data objects. Those objects are
# temporary container that's handy for this program, but is not meant to
# be reusable.
def build_iN_constcorr_prior(data, cmb=None, lknee0=2000, ivars=None, constcov=False):
	if ivars is None: ivars = data.ivars
	N    = enmap.zeros((data.n,data.n)+data.maps.shape[-2:], data.maps.wcs, data.maps.dtype)
	ref  = np.mean(ivars,(-2,-1))
	ref  = np.maximum(ref, np.max(ref)*1e-4)
	norm = 1/(ref/ivars.pixsize())
	for i, freq in enumerate(data.freqs):
		lknee  = lknee0*freq/100
		N[i,i] = (1+(np.maximum(0.5,data.l)/lknee)**-3.5) * norm[i]
	# Deconvolve the pixel window from the theoretical flat-at-high-l spectrum
	N  = N / data.wy[:,None]**2 / data.wx[None,:]**2
	# Apply the beam-convolved cmb if available
	if cmb is not None:
		Bf = data.fconvs[:,None,None]*data.beams
		N += cmb*Bf[:,None]*Bf[None,:]
		del Bf
	if not constcov:
		N /= norm[:,None,None,None]**0.5*norm[None,:,None,None]**0.5
	iN = array_ops.eigpow(N, -1, axes=[0,1])
	return iN

def build_iN_constcov_prior(data, cmb=None, lknee0=2000):
	return build_iN_constcorr_prior(data, cmb=cmb, lknee0=lknee0, constcov=True)

def build_iN_constcorr(data, maps, smooth="angular", brel=2, lsigma=500, lmin=100, ivars=None, constcov=False):
	if ivars is None: ivars = data.ivars
	#enmap.write_map("constcorr_imap1.fits", maps)
	if not constcov: maps = maps*ivars**0.5
	#enmap.write_map("constcorr_imap2.fits", maps)
	fhmap = enmap.map2harm(maps, spin=0, normalize="phys") / maps.pixsize()**0.5
	del maps
	N     = (fhmap[:,None]*np.conj(fhmap[None,:])).real
	N    /= np.maximum(data.fapod, 1e-8)
	del fhmap
	# Smooth in piwwin-space, since things are expected to be more isotopic there
	N = N * data.wy[:,None]**2 * data.wx[None,:]**2
	if smooth == "angular":
		N = smooth_ps_angular(N, brel=brel)
	elif smooth == "gauss":
		N = smooth_ps_gauss(N, lsigma=lsigma)
	elif smooth == "mixed":
		N = smooth_ps_mixed(N, brel=brel, lsigma=lsigma)
	else:
		raise ValueError("Unrecognized smoothing '%s'" % str(smooth))
	#enmap.write_map("N.fits", N)
	iN = analysis.safe_pow(N, -1)
	#enmap.write_map("iN.fits", iN)
	iN = iN * data.wy[:,None]**2 * data.wx[None,:]**2
	iN[:,:,data.l<lmin] = 0
	return iN

def build_iN_constcov(data, maps, smooth="isotropic", brel=2, lsigma=500, lmin=100):
	return build_iN_constcorr(data, maps, smooth=smooth, brel=brel, lsigma=lsigma, lmin=lmin, constcov=True)

def build_wiN(maps, wt, smooth=5):
	ncomp  = len(maps)
	wnoise = wt.map2wave(maps)
	wiN    = multimap.zeros([geo.with_pre((ncomp,ncomp)) for geo in wnoise.geometries], dtype=maps.dtype)
	for i, m in enumerate(wnoise.maps):
		srad = 2*np.pi/wt.basis.lmaxs[i]*smooth
		Nmap = enmap.smooth_gauss(m[:,None]*m[None,:], srad)
		wiN.maps[i] = analysis.safe_pow(Nmap, -1)
	return wiN

#def build_wiN_ivar(maps, ivars, wt, smooth=5, tol=1e-4):
#	ncomp  = len(maps)
#	wnoise = wt.map2wave(maps)
#	wivar  = wt.map2wave(ivars, half=True)
#	wiN    = multimap.zeros([geo.with_pre((ncomp,ncomp)) for geo in wnoise.geometries], dtype=maps.dtype)
#	for i, (m, iv) in enumerate(zip(wnoise.maps, wivar.maps)):
#		# Want to estimate smooth behavior with wivar as weight
#		srad = 2*np.pi/wt.basis.lmaxs[i]*smooth
#		rhs  = enmap.smooth_gauss(m[:,None]*m[None,:]*iv, srad)
#		div  = enmap.smooth_gauss(iv, srad)
#		div  = np.maximum(div, np.max(div,(-2,-1))[:,None,None]*tol)
#		Nmap = rhs/div
#		del rhs, div
#		wiN.maps[i] = safe_pow(Nmap, -1)
#	return wiN

def build_wiN_ivar(maps, ivars, wt, apod=None, smooth=5, tol=1e-2):
	# This is very ad-hoc, but it mostly works now. It's hard to
	# avoid things blowing up near the edge, but at least that's
	# only at the level of a few thousand fake sources at the edge now,
	# not infinitely bright spots with huge ringing around them.
	ncomp  = len(maps)
	wnoise = wt.map2wave(maps)
	wivar  = wt.map2wave(ivars, half=True)
	if apod is not None: wapod = wt.map2wave(apod, half=True)
	wiN    = multimap.zeros([geo.with_pre((ncomp,ncomp)) for geo in wnoise.geometries], dtype=maps.dtype)
	for i, (m, iv) in enumerate(zip(wnoise.maps, wivar.maps)):
		hiv = np.maximum(iv,0)**0.5
		# We model the noise cov per wavelet scale as v**0.5 W v**0.5.
		# Since these are pixel-diagonal, that's just W v.
		# Measure W from whitened map. Let's still use ivar-weighting.
		weight = hiv[:,None]*hiv[None,:]
		srad = 2*np.pi/wt.basis.lmaxs[i]*smooth
		rhs  = enmap.smooth_gauss(m[:,None]*m[None,:]*hiv[:,None]*hiv[None,:]*weight, srad)
		div  = enmap.smooth_gauss(weight, srad)*hiv[:,None]*hiv[None,:]
		norm = np.max(np.einsum("aa...->a...",div),(-2,-1))[:,None,None]**0.5
		div  = np.maximum(div, norm[:,None]*norm[None,:]*tol)
		#enmap.write_map("rhs_%02d.fits" % i, rhs)
		#enmap.write_map("div_%02d.fits" % i, div)

		Nmap = rhs/div
		# Don't allow the noise to be lower than the white noise floor
		for j in range(len(iv)):
			Nmap[j,j] = np.maximum(Nmap[j,j], 1/np.maximum(iv[j],1e-20))
		del rhs, div
		# Too low regions should get no weight
		#enmap.write_map("Nmap_%02d.fits" % i, Nmap)
		wiN.maps[i]  = analysis.safe_pow(Nmap, -1)
		if apod is not None:
			# Compute how much our result has been biased by apodized/missing regions.
			# I don't understand why it didn't work to bake this into the
			# equation above.
			norm  = np.maximum(np.max(wapod.maps[i],(-2,-1)), 1)
			bias  = enmap.smooth_gauss(wapod.maps[i]/norm[:,None,None], srad)
			bias *= bias > 0.1
			# Overcompensate for this
			wiN.maps[i] *= bias[:,None]*bias[None,:]
		#enmap.write_map("iNmap_%02d.fits" % i, wiN.maps[i])
		#enmap.write_map("wapod_%02d.fits" % i, wapod.maps[i]/np.max(wapod.maps[i]))
	return wiN

def build_foreground_var(maps, bsize1=8, bsize2=4, scale=0.5):
	"""Given maps[ncomp,ny,nx] that may contain point sources etc., return
	an estimate of the foreground variance. This can be used to downweight
	high-foreground regions."""
	vbox = np.array([[-32,-88],[-31,-90]])*utils.degree
	# Can't use op=np.var here because downgrade does the operation in two steps
	v1  = enmap.downgrade(maps**2, bsize1, op=np.mean,   inclusive=True)
	v1 -= enmap.downgrade(maps,    bsize1, op=np.mean,   inclusive=True)**2
	v2  = enmap.downgrade(v1,      bsize2, op=np.median, inclusive=True)
	for v in v2: v[v==0] = np.median(v[v!=0])
	res = np.exp(enmap.project(np.log(v2), maps.shape, maps.wcs, order=1))
	res *= scale
	return res

def apply_foreground_var(ivars, fg_var, tol=1, exp=2):
	"""Given ivar and fg_var, combine them into a full ivar which is extra
	penalizing for high-fg areas"""
	ref = np.mean(ivars,(-2,-1))
	ref[ref<=0] = 1
	var  = 1/np.maximum(ivars, ref[:,None,None]*1e-10)
	var *= np.maximum(1, tol*(fg_var/(var*tol))**exp)
	return 1/var

def sim_noise_oneoverf(ivar, apod, lknee=3000, lmin=50, alpha=-3.5):
	l      = np.maximum(ivar.modlmap(), lmin)
	fscale = (1+(l/lknee)**alpha)**0.5
	ref    = np.mean(ivar[ivar!=0])
	sim    = enmap.rand_gauss(ivar.shape, ivar.wcs, dtype=ivar.dtype)
	sim    = enmap.ifft(enmap.fft(sim)*fscale).real * np.maximum(ivar, ref*1e-3)**-0.5 * apod**2
	return sim

def read_data(fnames, sel=None, pixbox=None, box=None, geometry=None, comp=0, split=0, unit="flux", dtype=np.float32,
		beam_rmax=5*utils.degree, beam_res=2*utils.arcsec, deconv_pixwin=True, apod=15*utils.arcmin, mask=None,
		ivscale=[1,0.5,0.5]):
	"""Read multi-frequency data for a single split of a single component, preparing it for
	analysis."""
	# Read in our data files and harmonize
	br   = np.arange(0,beam_rmax,beam_res)
	data = bunch.Bunch(maps=[], ivars=[], beams=[], freqs=[], l=None, bls=[], names=[], beam_profiles=[])
	for ifile in fnames:
		d = mapdata.read(ifile, sel=sel, pixbox=pixbox, box=box, geometry=geometry)
		# The 0 here is just selecting the first split. That is, we don't support splits
		data.maps .append(d.maps [split].astype(dtype)[comp])
		data.ivars.append(d.ivars[split].astype(dtype)*ivscale[comp])
		data.freqs.append(d.freq)
		if data.l is None: data.l = d.maps[0].modlmap()
		data.beams.append(enmap.ndmap(np.interp(data.l, np.arange(len(d.beam)), d.beam/np.max(d.beam)), d.maps[0].wcs).astype(dtype))
		data.names.append(".".join(os.path.basename(ifile).split(".")[:-1]))
		data.bls.append(d.beam)
		data.beam_profiles.append(np.array([br, curvedsky.harm2profile(d.beam,br)]).astype(dtype))

	data.maps  = enmap.enmap(data.maps )
	data.ivars = enmap.enmap(data.ivars)
	data.beams = enmap.enmap(data.beams)
	data.freqs = np.array(data.freqs)
	if unit == "uK":
		data.fconvs = np.full(len(data.freqs), 1.0, dtype)
	elif unit == "flux":
		data.fconvs= (utils.dplanck(data.freqs*1e9, utils.T_cmb)/1e3).astype(dtype) # uK -> mJy/sr
	else: raise ValueError("Unrecognized unit '%s'" % str(unit))
	data.n     = len(data.freqs)

	# Apply the unit
	data.maps  *= data.fconvs[:,None,None]
	data.ivars /= data.fconvs[:,None,None]**2

	if mask is not None:
		mask_map    = 1-enmap.read_map(mask, sel=sel, pixbox=pixbox, box=box)
		data.ivars *= mask_map
		del mask_map

	# Should generalize this to handle internal map edges and frequency differences
	mask      = enmap.shrink_mask(enmap.grow_mask(data.ivars>0, 1*utils.arcmin), 1*utils.arcmin)
	apod_map  = enmap.apod_mask(mask, apod)
	data.apod = apod_map
	data.fapod= np.mean(apod_map**2)
	data.maps  *= apod_map
	data.ivars *= apod_map**2

	# Get the pixel window and optionall deconvolve it
	data.wy, data.wx = [w.astype(dtype) for w in enmap.calc_window(data.maps.shape)]
	if deconv_pixwin:
		data.maps = enmap.ifft(enmap.fft(data.maps)/data.wy[:,None]/data.wx[None,:]).real

	return data

def build_cmb_2d(shape, wcs, cl_cmb, dtype=np.float32):
	lmap = enmap.lmap(shape, wcs)
	l    = np.sum(lmap**2,0)**0.5
	cmb  = enmap.samewcs(utils.interp(l, np.arange(cl_cmb.shape[-1]), cl_cmb), l).astype(dtype)
	# Rotate [TEB,EB] -> [TQU,TQU]. FIXME: not a perfect match
	R = enmap.queb_rotmat(lmap, spin=2, inverse=True)
	cmb[1:,:] = np.einsum("abyx,bcyx->acyx", R, cmb[1:,:])
	cmb[:,1:] = np.einsum("abyx,cbyx->acyx", cmb[:,1:], R)
	return cmb

def build_case_ptsrc(data, scaling=None):
	if scaling is None: scaling = np.full(data.n, 1.0)
	scaling  = np.asarray(scaling).astype(data.maps.dtype)
	modeller = analysis.ModellerPerfreq(data.maps.shape, data.maps.wcs, data.beam_profiles)
	return bunch.Bunch(profile=data.beams, scaling=scaling, modeller=modeller)

def build_case_tsz(data, size=1*utils.arcmin, scaling=None):
	if scaling is None:
		scaling = utils.tsz_spectrum(data.freqs*1e9)/np.abs(utils.tsz_spectrum(data.freq0*1e9))
	# Get the fourier shapes
	lprofs  = (utils.tsz_tform(size, data.l)*data.beams).astype(data.maps.dtype)
	lprofs /= np.max(lprofs, (-2,-1))[:,None,None]
	# Get the real-space templates for the model
	profs1d = []
	for i in range(data.n):
		lprof1d  = utils.tsz_tform(size, np.arange(len(data.bls[i])))*data.bls[i]
		lprof1d /= np.max(lprof1d)
		br = data.beam_profiles[i][0]
		profs1d.append(np.array([br, curvedsky.harm2profile(lprof1d, br)]))
	modeller = analysis.ModellerScaled(data.maps.shape, data.maps.wcs, profs1d, scaling)
	return bunch.Bunch(profile=lprofs, scaling=scaling, modeller=modeller)

def build_nmat_prior(data, type="constcorr", cmb=None, pol=False, fg_var=None):
	if type == "constcov":
		iN    = build_iN_constcov_prior(data, cmb=cmb, lknee0=800 if pol else 2000)
		nmat  = analysis.NmatConstcov(iN, data.apod)
	elif type == "constcorr":
		ivars = data.ivars
		if fg_var is not None: ivars = apply_foreground_var(ivars, fg_var)
		iN    = build_iN_constcorr_prior(data, ivars=ivars, cmb=cmb, lknee0=800 if pol else 2000)
		nmat  = analysis.NmatConstcorr(iN, ivars)
	else:
		raise ValueError("Unsupported prior nmat: '%s'" % str(args.nmat1))
	return nmat

def build_nmat_empirical(data, noise_map, type="constcorr", smooth="angular", fg_var=None):
	if type == "constcov":
		iN    = build_iN_constcov(data, noise_map, smooth=smooth)
		nmat  = analysis.NmatConstcov(iN)
	elif type == "constcorr":
		ivars = data.ivars
		if fg_var is not None: ivars = apply_foreground_var(ivars, fg_var)
		iN    = build_iN_constcorr(data, noise_map, smooth=smooth, ivars=ivars)
		nmat  = analysis.NmatConstcorr(iN, ivars)
	elif type == "wavelet":
		from pixell import uharm
		wbasis= wavelets.ButterTrim(step=2**0.5, lmin=50)
		uht   = uharm.UHT(data.maps.shape, data.maps.wcs, mode="flat")
		wt    = wavelets.WaveletTransform(uht, basis=wbasis)
		wiN   = build_wiN(noise_map, wt)
		nmat  = analysis.NmatWavelet(wt, wiN)
	elif type == "weighted-wavelet":
		from pixell import uharm
		wbasis= wavelets.ButterTrim(step=2**0.5, lmin=50)
		uht   = uharm.UHT(data.maps.shape, data.maps.wcs, mode="flat")
		wt    = wavelets.WaveletTransform(uht, basis=wbasis)
		wiN   = build_wiN_ivar(noise_map, data.ivars, wt, apod=data.apod)
		nmat  = analysis.NmatWavelet(wt, wiN)
	else:
		raise ValueError("Unsupported empirical nmat: '%s'" % str(args.nmat1))
	return nmat

def find_objects(data, cases, nmat, snmin=5, resid=False, verbose=False):
	# Build an interative finder from a multi-case finder and a multi-case modeller
	raw_finder    = analysis.FinderMulti(nmat, cases.profile, cases.scaling, save_snr=True)
	modeller      = analysis.ModellerMulti(cases.modeller)
	finder        = analysis.FinderIterative(raw_finder, modeller)
	# Run the finder
	res           = finder(data.maps, snmin=snmin, verbose=verbose)
	# Add some useful quantities to the result object
	res.maps      = data.maps
	res.snr       = raw_finder.snr
	res.fconvs    = data.fconvs
	if resid:
		res.resid     = data.maps-res.model
		res.resid_snr = raw_finder(res.resid).snr
	return res

def measure_objects(data, cases, nmat, cat, resid=False, verbose=False):
	raw_measurer = analysis.MeasurerMulti([analysis.MeasurerSimple(nmat, profile, scaling) for profile, scaling in zip(cases.profile, cases.scaling)])
	modeller     = analysis.ModellerMulti(cases.modeller)
	measurer     = analysis.MeasurerIterative(raw_measurer, modeller)
	res          = measurer(data.maps, cat, verbose=verbose)
	# Add some useful quantities to the result object
	res.maps     = data.maps
	res.fconvs   = data.fconvs
	if resid:
		res.resid  = data.maps-res.model
	return res

def build_cases(data, templates):
	cases = []
	for params in templates:
		scaling = get_scaling(params, data.freqs, data.freq0)
		type    = params[0]
		if type == "ptsrc" or type == "graysrc":
			cases.append(build_case_ptsrc(data, scaling))
		elif type == "tsz":
			cluster_scale = params[1]*utils.arcmin
			cases.append(build_case_tsz(data, cluster_scale, scaling=scaling))
	cases = bunch.concatenate(cases)
	return cases

def get_scaling(params, freqs, freq0):
	# This is a separate function because we sometimes just want the scaling
	# without all the full beam and shape stuff
	type = params[0]
	if   type == "ptsrc":
		return (freqs/freq0)**params[1]
	elif type == "graysrc":
		return utils.graybody(freqs*1e9, params[1])/utils.graybody(freq0*1e9, params[1])
	elif type == "tsz":
		return utils.tsz_spectrum(freqs*1e9)/np.abs(utils.tsz_spectrum(freq0*1e9))

def inject_objects(data, cases, cat):
	modeller = analysis.ModellerMulti(cases.modeller)
	data.maps += modeller(cat)

# Yuck! I need a better solution for the catalogues!
def slice_cat_comp(cat, comp):
	odtype = []
	for sub in np.dtype(cat.dtype).descr:
		if sub[0] in ["flux_tot", "dflux_tot", "snr", "flux","dflux"]:
			sub = (sub[0], sub[1], sub[2][:-1])
		odtype.append(sub)
	ocat = np.zeros(len(cat), odtype).view(np.recarray)
	for key in ocat.dtype.names:
		ocat[key] = cat[key][...,comp] if key in["flux_tot", "dflux_tot", "snr", "flux", "dflux"] else cat[key]
	return ocat

default_templates = [("ptsrc",-0.66), ("ptsrc",0),("graysrc",10), ("tsz",0.1),
		("tsz",2), ("tsz",4), ("tsz",6), ("tsz",8)]
def search_maps(ifiles, mode="find", icat=None, sel=None, pixbox=None, box=None, templates=default_templates,
		cl_cmb=None, freq0=98.0, nmat1="constcorr", nmat2="constcorr", snr1=5, snr2=4, comps="TQU",
		dtype=np.float32, apod=15*utils.arcmin, verbose=False, sim_cat=None, sim_noise=False, mask=None):
	"""Search the maps given by ifiles for objects, returning a bunch containing a catalog
	of objects, etc.

	Arguments:
	* ifiles: A list of paths to mapdata files. These should be in µK units.
	* mode:   What operation to do. Either "find" or "fit". Defaults to "find"
	  * "find":  Do a blind object search in the maps.
	  * "fit":   Do forced photometry on the positions provided in the input catalog icat.
	* sel, pixbox, box: These let you work with a subset of the maps. Same meaning as
	  in enmap.read_map. Default to None.
	* templates: What spectral and spatial shapes to look for. This is a list of tuples
	  that will be passed to build_cases. Defaults to default_templates.
	* cl_cmb: The CMB angular power spectrum. Ideally [TQU,TQU,nl] (note: not TEB).
	  Used to build the blind noise model.
	* freq0: The reference frequency in GHz. This is used when reporting the overall flux
	  of multifrequency templates.
	* nmat1: The noise model to use for the first pass of the search. These are
	  built from simple analytic models, not measured from the data.
	  "constcov": A constant covariance noise model. This is fast, but
	    purely harmonic, so it can't handle variations in hitcount.
	  "constcorr": A constant correlation noise model, where a constant
	    noise spectrum is modulated by the hitcount. Handles both correlations
	    and spatial variations, in a limited way. [default]
	* nmat2: The noise model to use for the second pass of the search. These
	  are built from maps cleaned using the first pass. Defaults to "constcorr".
	  "none": Disables the second pass, returning the catalog found in the first pass
	  "constcov", "constcorr": As nmat1, but measured from the data. The power spectra
	  are smoothed isotropically for now.
	* snr1: The S/N threshold used for the first pass in "find" mode. Defaults to 5.
	* snr2. The S/N threshold used for the second pass in "find" mode. Defaults to 4.
	* comps: Controls the Stokes parameters considered. Can be "T" or "TQU". Defaults to "TQU".
	* dtype: The maps will be cast to this data type after reading in. Defaults to np.float32.
	* apod: How much apodization is used at the edges (including edges of unhit areas),
	  in radians. This is necessary to avoid ringing artifacts in the fourier transforms.
	  Defaults to 15 arcmin.
	* verbose: Whether to print what it's doing. Defaults to False.
	* sim_noise: Whether to replace the raw data with noise. Currently a cmb-less constcorr
	  realization. Should fix this.
	* sim_cat: A catalog to inject into the maps before the search. Defaults to None.
	* mask: Path to a mask enmap that's True in bad regions and False in good regions.
	  Bad regions will be given zero weight and apodized.
	
	Returns a bunch with the following members:
	* cat: A catalog with the data type [("ra", "d"), ("dec", "d"), ("snr", "d", (ncomp,)),
	  ("flux_tot", "d", (ncomp,)), ("dflux_tot", "d", (ncomp,)), ("flux", "d", (nfield,ncomp)),
	  ("dflux", "d", (nfield,ncomp)), ("case", "i"), ("contam", "d", (nfield,))]
	* maps: The maps that were searched [nfreq,ncomp,ny,nx]
	* model: The best-fit model
	* snr:  The S/N ratio for the input maps [ny,nx]
	* resid_snr: The S/N ratio after subtracting the best-fit model
	* freqs: The frequencies, in GHz
	* fconvs: The µK -> mJy/sr flux conversion factor for each frequency
	* inds: The index of each entry in the output catalog in the input catalog.
	  Only relevant when mode == "fit".
"""

	# Read in the total intensity data
	if verbose: print("Reading T from %s" % str(ifiles))
	data   = read_data(ifiles, sel=sel, pixbox=pixbox, box=box, dtype=dtype, apod=apod, mask=mask)
	data.freq0 = freq0
	ncomp  = len(comps)
	nfield = len(data.maps)
	cat_dtype  = [("ra", "d"), ("dec", "d"), ("snr", "d", (ncomp,)), ("flux_tot", "d", (ncomp,)),
			("dflux_tot", "d", (ncomp,)), ("flux", "d", (nfield,ncomp)), ("dflux", "d", (nfield,ncomp)),
			("case", "i"), ("contam", "d", (nfield,))]
	cases = build_cases(data, templates)

	# Get the part of the catalog inside our area
	if mode == "fit":
		inds    = np.where(np.any(data.ivars.at([icat.dec,icat.ra], order=0)>0,0))[0]
		subicat = icat[inds]
	else:
		inds    = None

	# Abort if we have no data to process
	if np.all(data.ivars == 0):
		map_tot = enmap.zeros((nfield,ncomp)+data.maps.shape[-2:], data.maps.wcs, dtype)
		cat     = np.zeros(0, cat_dtype)
		return bunch.Bunch(cat=cat, maps=map_tot, model=map_tot, snr=map_tot[0,0],
			resid_snr=map_tot[0,0], hits=map_tot[0,0], fconvs=data.fconvs, freqs=data.freqs, inds=inds)

	# Build our noise model, based on a 1/l spectrum + cmb + a foreground penalty
	cmb    = build_cmb_2d(*data.maps.geometry, cl_cmb, dtype=data.maps.dtype) if cl_cmb is not None else None
	fg_var = build_foreground_var(data.maps)
	nmat   = build_nmat_prior(data, type=nmat1, fg_var=fg_var, cmb=cmb[0,0] if cmb is not None else None)

	# Optionally inject signal. FIXME: This should be moved after nmat is defined,
	# so we can let nmat handle the noise simulation
	if sim_noise: data.maps = nmat.simulate()*data.apod
	if sim_cat is not None: inject_objects(data, cases, slice_cat_comp(sim_cat, 0))

	# Total intensity
	if mode == "find":
		if verbose: print("1st pass T find")
		res_t  = find_objects(data, cases, nmat, snmin=snr1, resid=nmat2=="none", verbose=verbose)
	elif mode == "fit":
		if verbose: print("1st pass T measure")
		res_t  = measure_objects(data, cases, nmat, slice_cat_comp(subicat, 0), resid=nmat2=="none", verbose=verbose)
	else: raise ValueError("Unrecognized mode '%s'" % (mode))
	if nmat2 != "none":
		noise  = data.maps-res_t.model
		if "bad_mask" in res_t:
			noise_apod = enmap.apod_mask(1-res_t.bad_mask, 10*utils.arcmin, edge=False)
			noise     *= noise_apod
			noise     /= np.mean(noise_apod**2)
			#enmap.write_map("noise.fits", noise)
			del noise_apod
		nmat   = build_nmat_empirical(data, noise, fg_var=fg_var, type=nmat2)
		if mode == "find":
			if verbose: print("2nd pass T find")
			res_t  = find_objects(data, cases, nmat, snmin=snr2, resid=True, verbose=verbose)
		elif mode == "fit":
			if verbose: print("2nd pass T measure")
			res_t  = measure_objects(data, cases, nmat, slice_cat_comp(subicat, 0), resid=True, verbose=verbose)
		else: raise ValueError("Unrecognized mode '%s'" % (mode))

	res = [res_t]
	# Polarization is always "fit", since anything that would be found in polarization
	# would definitely be found in total intensity
	if comps == "T":
		pass
	elif comps == "TQU":
		# Measure polarization too
		for comp in [1,2]:
			if verbose: print("Reading %s from %s" % (comps[comp], str(ifiles)))
			data  = read_data(ifiles, sel=sel, pixbox=pixbox, box=box, comp=comp, apod=apod)
			data.freq0 = freq0
			# Optionally inject signal
			if sim_cat is not None: inject_objects(data, cases, slice_cat_comp(sim_cat, comp))
			if verbose: print("1st pass %s measure" % comps[comp])
			nmat  = build_nmat_prior(data, type=nmat1, pol=True, cmb=cmb[comp,comp] if cmb is None else None)
			res_p = measure_objects(data, cases, nmat, res_t.cat, verbose=verbose)
			if nmat2 != "none":
				if verbose: print("2nd pass %s measure" % comps[comp])
				nmat  = build_nmat_empirical(data, noise_map=data.maps-res_p.model, type=nmat2)
				res_p = measure_objects(data, cases, nmat, res_t.cat, verbose=verbose)
			res.append(res_p)
	# First the catalog
	cat = np.zeros(len(res_t.cat), cat_dtype).view(np.recarray)
	cat.ra     = res_t.cat.ra
	cat.dec    = res_t.cat.dec
	cat.case   = res_t.cat.case
	cat.contam = res_t.cat.contam
	for i in range(len(res)):
		cat.snr[:,i]       = res[i].cat.snr
		cat. flux_tot[:,i] = res[i].cat. flux_tot
		cat.dflux_tot[:,i] = res[i].cat.dflux_tot
		cat. flux[:,:,i]   = res[i].cat. flux
		cat.dflux[:,:,i]   = res[i].cat.dflux
	# Then the maps
	map_tot   = enmap.samewcs(np.concatenate([r.maps [:,None] for r in res],1), data.maps)
	model_tot = enmap.samewcs(np.concatenate([r.model[:,None] for r in res],1), data.maps)
	result = bunch.Bunch(cat=cat, maps=map_tot, model=model_tot, fconvs=data.fconvs, freqs=data.freqs, inds=inds)
	# These only exist in "find" mode
	for key in ["snr", "resid_snr", "hits"]:
		result[key] = res_t[key] if key in res_t else None
	return result

def search_maps_tiled(ifiles, odir, tshape=(1000,1000), margin=100, padding=150,
		mode="find", icat=None, box=None, pixbox=None, sel=None, mask=None,
		templates=default_templates, cl_cmb=None, freq0=98.0, nmat1="constcorr",
		nmat2="constcorr", snr1=5, snr2=4, comps="TQU", dtype=np.float32, comm=None,
		cont=False, sim_cat=None, sim_noise=False, verbose=False):
	wdir = odir + "/work"
	utils.mkdir(wdir)
	if comm is None: comm = bunch.Bunch(rank=0, size=1)
	tshape = np.zeros(2,int)+tshape
	meta   = mapdata.read_meta(ifiles[0])
	# Allow us to slice the map that will be tiled
	geo    = enmap.Geometry(*meta.map_geometry)
	if pixbox is not None or box is not None:
		geo  = geo.submap(pixbox=pixbox, box=box)
	if sel is not None: geo = geo[sel]
	shape  = np.array(geo.shape[-2:])
	ny,nx  = (shape+tshape-1)//tshape
	def is_done(ty,tx): return os.path.isfile("%s/cat_%03d_%03d.fits" % (wdir, ty,tx))
	tyxs   = [(ty,tx) for ty in range(ny) for tx in range(nx) if (not cont or not is_done(ty,tx))]
	for ind in range(comm.rank, len(tyxs), comm.size):
		# Get basic area of this tile
		tyx = np.array(tyxs[ind])
		if verbose: print("%2d Processing tile %2d %2d of %2d %2d" % (comm.rank, tyx[0], tyx[1], ny, nx))
		yx1 = tyx*tshape
		yx2 = np.minimum((tyx+1)*tshape, shape)
		# Apply padding
		wyx1 = yx1-margin-padding
		wyx2 = yx2+margin+padding
		# Transform from box-relative pixbox to global pixbox
		off  = enmap.pixbox_of(meta.map_geometry[1], *geo)[0]
		wyx1 += off
		wyx2 += off
		# Process this tile
		res = search_maps(ifiles, mode=mode, icat=icat, pixbox=[wyx1,wyx2],
				templates=templates, mask=mask, cl_cmb=cl_cmb, freq0=freq0,
				nmat1=nmat1, nmat2=nmat2, snr1=snr1, snr2=snr2, comps=comps,
				dtype=dtype, sim_cat=sim_cat, sim_noise=sim_noise, verbose=verbose)
		# Write tile results to work directory. We do this to avoid using too much memory,
		# and to allow us to continue
		write_results(wdir, res, padding=padding, tag="%03d_%03d" % tuple(tyx))
	comm.Barrier()
	# When everything's done, merge things into single files
	if comm.rank == 0:
		merge_results(wdir, odir, geo, tshape=tshape, margin=margin, verbose=verbose)

def merge_tiled_cats(cats, boxes, margin=2*utils.arcmin):
	"""Merge the list of catalogs cats, where each catalog owns the corresponding
	bounding box in boxes, but also also an overlapping area around. For the area
	select(box,0,edge-margin) the catalog will be used directly. For the area
	select(box,edge-margin,edge+margin) duplicates will be looked for and removed."""
	# First get the central part of each catalog, which can be used as they are
	boxes = np.asarray(boxes) # [nbox,{from,to},{dec,ra}]
	boxes_inner = np.concatenate([boxes[:,None,0,:]+margin,boxes[:,None,1,:]-margin],1)
	boxes_outer = np.concatenate([boxes[:,None,0,:]-margin,boxes[:,None,1,:]+margin],1)
	cats_inner  = []
	cats_border = []
	for ci, cat in enumerate(cats):
		ref_ra = np.mean(boxes[ci,:,1])
		pos    = np.array([cat.dec,utils.rewind(cat.ra, ref_ra)]).T
		inner  = np.all(pos > boxes_inner[ci][0],1) & np.all(pos < boxes_inner[ci][1],1)
		outer  = np.any(pos < boxes_outer[ci][0],1) | np.any(pos > boxes_outer[ci][1],1)
		border = ~inner & ~outer
		cats_inner .append(cat[inner ])
		cats_border.append(cat[border])
	cat_inner  = np.concatenate(cats_inner)
	if len(cats_border) > 0 and "inds" in cats_border[0].dtype.names:
		cat_border = merge_duplicates_inds(cats_border)
	else:
		cat_border = merge_duplicates(cats_border)
	ocat   = np.concatenate([cat_inner, cat_border]).view(np.recarray)
	order  = np.argsort(ocat.snr[:,0])[::-1]
	ocat   = ocat[order]
	return ocat

def merge_duplicates(cats, dr=1*utils.arcmin, dsnr=1.5):
	"""Given a list of catalogs that could contain duplicates, choose one result
	for each dr*dr cell spatially and log-step dsnr in S/N ratio."""
	from scipy import spatial
	cat  = np.concatenate(cats).view(np.recarray)
	lsnr = np.log(np.maximum(np.abs(cat.snr[:,0]),1))/np.log(dsnr)
	pos  = utils.ang2rect([cat.ra,cat.dec]).T/dr
	x    = np.concatenate([pos,lsnr[:,None]],1)
	tree = spatial.cKDTree(x)
	groups = tree.query_ball_tree(tree, 1)
	done = np.zeros(len(cat),bool)
	ocat = []
	for gi, group in enumerate(groups):
		# Remove everything that's done
		group = np.array(group)
		group = group[~done[group]]
		if len(group) == 0: continue
		# Keep the first entry
		first = group[0]
		ocat.append(cat[first])
		# Mark all entries as done. Assume hte rest were duplicates
		done[group] = True
	ocat = np.array(ocat, cats[0].dtype).view(np.recarray)
	return ocat

def merge_duplicates_inds(cats):
	"""Given a list of catalogs with object ids in the field "inds",
	return the last such field encountered"""
	print("cats[0].dtype", cats[0].dtype)
	all_inds = utils.union([cat.inds for cat in cats])
	ocat     = np.zeros(len(all_inds), cats[0].dtype).view(np.recarray)
	for cat in cats:
		rel_inds = utils.find(all_inds, cat.inds)
		ocat[rel_inds] = cat
	return ocat

def make_edge_interp(n, w, left=True, right=True, dtype=np.float32):
	x = (np.arange(2*w)+1).astype(dtype)/(2*w+1)
	l = x       if left  else x*0+1
	r = x[::-1] if right else x*0+1
	m = np.full(n-2*w, 1.0, dtype)
	return np.concatenate([l,m,r])

def merge_tiles(shape, wcs, paths, margin=100, dtype=np.float32):
	# Get the pre-dimensions from the first tile
	shape = enmap.read_map_geometry(paths[0][0])[0][:-2]+shape[-2:]
	omap  = enmap.zeros(shape, wcs, dtype)
	ny    = len(paths)
	nx    = len(paths[0])
	for ty in range(ny):
		for tx in range(nx):
			fname = paths[ty][tx]
			imap  = enmap.read_map(fname)
			h, w  = imap.shape[-2:]
			wy    = make_edge_interp(h-2*margin, margin, ty>0, ty<ny-1, dtype=imap.dtype)
			wx    = make_edge_interp(w-2*margin, margin,                dtype=imap.dtype)
			imap  = imap * wy[:,None] * wx[None,:]
			omap.insert(imap, op=np.ndarray.__iadd__)
	return omap

def write_results(odir, res, padding=0, tag=None):
	def unpad(map):
		if padding == 0: return map
		else: return map[...,padding:-padding,padding:-padding]
	def fix(map): return unpad(enmap.apply_window(map))/res.fconvs[:,None,None,None]
	utils.mkdir(odir)
	suffix = "" if tag is None else "_" + tag
	enmap.write_map("%s/map%s.fits"       % (odir,suffix), fix(res.maps))
	enmap.write_map("%s/model%s.fits"     % (odir,suffix), fix(res.model))
	enmap.write_map("%s/resid%s.fits"     % (odir,suffix), fix(res.maps-res.model))
	if res.snr is not None:
		enmap.write_map("%s/map_snr%s.fits"   % (odir,suffix), unpad(res.snr))
	if res.resid_snr is not None:
		enmap.write_map("%s/resid_snr%s.fits" % (odir,suffix), unpad(res.resid_snr))
	# If we have indices of the catalog objects into a predefined catalog, then
	# append that as a new field, so we can use it in merge_results later
	cat = recfunctions.append_fields(res.cat, "inds", res.inds) if res.inds is not None else res.cat
	pointsrcs.write_sauron("%s/cat%s.fits"% (odir,suffix), cat)

def merge_results(wdir, odir, geo, tshape=(1000,1000), margin=100, verbose=False):
	if verbose: print("Reducing")
	shape, wcs = geo
	shape = np.array(shape[-2:])
	ny,nx = (shape+tshape-1)//tshape
	# Get the tile catalogs and their area of responsibility
	cats = []; boxes = []
	for ty in range(ny):
		for tx in range(nx):
			tyx    = np.array([ty,tx])
			pixbox = np.array([tyx*tshape, np.minimum((tyx+1)*tshape, shape)])
			boxes.append(np.sort(enmap.pixbox2skybox(*geo, pixbox),0))
			cats .append(pointsrcs.read_sauron(wdir + "/cat_%03d_%03d.fits" % (ty,tx)))
	if verbose: print("Merging %d cats" % len(cats))
	cat = merge_tiled_cats(cats, boxes)
	cat.ra = utils.rewind(cat.ra)
	pointsrcs.write_sauron("%s/cat.fits" % odir, cat)
	pointsrcs.write_sauron("%s/cat.txt"  % odir, cat)
	for name in ["map", "model", "resid", "map_snr", "resid_snr"]:
		paths = [["%s/%s_%03d_%03d.fits" % (wdir,name,ty,tx) for tx in range(nx)] for ty in range(ny)]
		if not os.path.isfile(paths[0][0]): continue
		if verbose: print("Merging tiles for %s" % name)
		dtype = enmap.read_map(paths[0][0]).dtype
		map   = merge_tiles(*geo, paths, dtype=dtype, margin=margin)
		enmap.write_map("%s/%s.fits" % (odir,name), map)
		del map
