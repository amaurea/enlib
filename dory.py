from __future__ import division, print_function
import numpy as np, os, time, sys
from scipy import ndimage, spatial, integrate
from . import enmap, utils, bunch, mpi, fft, bench, pointsrcs

cat_dtype = [("ra","f"),("dec","f"),("amp","3f"),("damp","3f"),("flux","3f"),("dflux","3f"),("npix","f"),("status","i")]

def get_beam(fname):
	try:
		sigma = float(fname)*utils.arcmin*utils.fwhm
		l     = np.arange(40e3)
		beam  = np.exp(-0.5*(l*sigma)**2)
	except ValueError:
		beam = np.loadtxt(fname, usecols=(1,))
	return beam

def read_boxes_ds9(fname):
	boxes = []
	with open(fname, "r") as ifile:
		for line in ifile:
			if line.startswith("box("):
				toks = line[4:-1].split(",")
				ra   = float(toks[0])*utils.degree
				dec  = float(toks[1])*utils.degree
				wra  = float(toks[2][:-1])*utils.arcsec
				wdec = float(toks[3][:-1])*utils.arcsec
				boxes.append([[dec-wdec/2,ra+wra/2],[dec+wdec/2,ra-wra/2]])
	boxes = np.array(boxes)
	return boxes

def read_boxes_txt(fname):
	boxes  = np.loadtxt(fname)[:,:4]
	boxes  = np.transpose(boxes.reshape(-1,2,2),(0,2,1))[:,:,::-1]
	boxes *= utils.degree
	return boxes

def get_regions(regfile, shape, wcs):
	if regfile is None: regfile = "full"
	toks = regfile.split(":")
	name, args = toks[0], toks[1:]
	if name == "full":
		# The simplest possible region, covering the whole patch
		regions = np.array([[0,0],shape[-2:]])[None]
	elif name == "tile":
		tsize = np.array([int(a) for a in args[:2]])
		if   tsize.size == 0: tsize = np.array([480,480])
		elif tsize.size == 1: tsize = np.array([tsize[0],tsize[0]])
		regions = [[[y,x],[y+tsize[0],x+tsize[1]]] for y in range(0, shape[-2], tsize[0]) for x in range(0, shape[-1], tsize[1])]
		regions = np.array(regions)
	elif name == "box":
		# Specify boxes directly on the command line. Not the most elegant syntax. dec1:ra1:dec2:ra2
		boxes   = np.array([float(w) for w in toks[1:5]]).reshape(-1,2,2)*utils.degree
		regions = np.array([enmap.skybox2pixbox(shape, wcs, box) for box in boxes])
		regions = np.round(regions).astype(int)
	elif name == "adaptive":
		# This one would use a low-res div and build regions with reasonably similar hitcounts
		# without having them become too large, too small or too empty. This is a pretty difficult
		# problem.
		raise NotImplementedError("Adaptive region splitting is not implemented yet")
	elif os.path.isfile(regfile):
		# Read explicit regions from file
		try: boxes = read_boxes_txt(regfile)
		except ValueError: boxes = read_boxes_ds9(regfile)
		# And turn them into pixel bounding boxes
		regions = np.array([enmap.skybox2pixbox(shape, wcs, box) for box in boxes])
		regions = np.round(regions).astype(int)
	else:
		raise ValueError("Unrecognized region type '%s'" % regfile)
	return regions

def split_regions(regions, size):
	"""Split regions into the maximum number of subregions such that each region
	is at least size pixels"""
	oregions = []
	for ((y1,x1),(y2,x2)) in regions:
		sy, sx = size, size
		h, w = (y2-y1), (x2-x1)
		# If regions are too short/thin make them taller/wider to compensate
		if h < sy and w < sx:
			oregions.append([[y1,x1],[y2,x2]])
		if   h < sy: sx = (sx*sy)//h
		elif w < sx: sy = (sy*sx)//w
		# We want even splits
		ny = (h+sy)//sy
		nx = (w+sx)//sx
		# Do the actual splitting
		for by in range(ny):
			oy1 = y1 + by*h//ny
			oy2 = y1 + (by+1)*h//ny
			for bx in range(nx):
				ox1 = x1 + bx*w//nx
				ox2 = x1 + (bx+1)*w//nx
				#print("h: %d w: %d sy: %d sx: %d th: %d tw: %d" % (h,w,sy,sx,oy2-oy1, ox2-ox1))
				oregions.append([[oy1,ox1],[oy2,ox2]])
	return np.array(oregions).astype(int)

def pad_region(region, pad, fft=False):
	region = np.array(region)
	region[...,0,:] -= pad
	region[...,1,:] += pad
	if fft: region = pad_region_fft(region)
	return region

def pad_region_fft(region):
	region = np.array(region)
	ndim = region.shape[-1]
	for rflat in region.reshape(-1,2,ndim):
		for i in range(ndim):
			size = rflat[1,i]-rflat[0,i]
			rflat[1,i] = rflat[1,i] + fft.fft_len(size, "above") - size
	return region

def get_apod_holes(div, pixrad):
	return enmap.samewcs(0.5*(1-np.cos(np.pi*np.minimum(1,ndimage.distance_transform_edt(div>0)/float(pixrad)))))

def smooth_ps_gauss(ps, lsigma):
	"""Smooth a 2d power spectrum to the target resolution in l. Simple
	gaussian smoothing avoids ringing."""
	# First get our pixel size in l
	ly, lx = enmap.laxes(ps.shape, ps.wcs)
	ires   = np.array([ly[1],lx[1]])
	sigma_pix = np.abs(lsigma/ires)
	fmap  = enmap.fft(ps)
	ky    = np.fft.fftfreq(ps.shape[-2])*sigma_pix[0]
	kx    = np.fft.fftfreq(ps.shape[-1])*sigma_pix[1]
	kr2   = ky[:,None]**2+kx[None,:]**2
	fmap *= np.exp(-0.5*kr2)
	return enmap.ifft(fmap).real

def safe_mean(arr, bsize=100):
	arr   = arr.reshape(-1)
	nblock = arr.size//bsize
	if nblock <= 1: return np.mean(arr)
	means = np.mean(arr[:nblock*bsize].reshape(nblock,bsize),-1)
	means = np.concatenate([means,[np.mean(arr[(nblock-1)*bsize:])]])
	return np.median(means)

def get_snmap_norm(snmap, bsize=240):
	norm = snmap*0+1
	ny, nx = np.array(snmap.shape[-2:])//bsize
	for by in range(ny):
		y1 = by*bsize
		y2 = (by+1)*bsize if by < ny-1 else snmap.shape[-2]
		for bx in range(nx):
			x1 = bx*bsize
			x2 = (bx+1)*bsize if bx < nx-1 else snmap.shape[-1]
			sub  = snmap[y1:y2,x1:x2]
			vals = sub[sub!=0]
			if vals.size == 0: continue
			std  = safe_mean(vals**2)**0.5
			norm[y1:y2,x1:x2] = std
	return norm

def measure_noise(noise_map, margin=15, apod=15, ps_res=200):
	# Ignore the margin and apodize the rest, while keeping the same overall shape
	apod_map  = enmap.extract((noise_map[margin:-margin,margin:-margin]*0+1).apod(apod), noise_map.shape, noise_map.wcs)
	noise_map = noise_map*apod_map
	ps        = np.abs(enmap.fft(noise_map))**2
	# Normalize to account for the masking
	ps /= np.mean(apod_map**2)
	#enmap.write_map("ps1.fits", ps*0+np.fft.fftshift(ps))
	# This smoothing is not optimal. It will overestimate the nosie at high l
	# and underestimate it at low l for red spectra like the atmosphere. Might
	# consider importing the more fancy smoothing from jointmap, or dividing out
	# the radial part first.
	ps     = smooth_ps_gauss(ps, ps_res)
	#enmap.write_map("ps2.fits", ps*0+np.fft.fftshift(ps))
	return ps

def planck_hack(ps2d, lcut=6000):
	ops2d= ps2d.copy()
	l    = ps2d.modlmap()
	ref  = np.mean(ps2d[(l<lcut)&(l>lcut*0.9)])
	ops2d[l>lcut] = ref
	return ops2d

def calc_2d_beam(beam1d, shape, wcs):
	lmap   = enmap.modlmap(shape, wcs)
	beam2d = enmap.ndmap(np.interp(lmap, np.arange(len(beam1d)), beam1d),wcs)
	return beam2d

def build_filter(ps, beam2d):
	# Build our matched filter, assumping beam-shaped point sources
	filter = beam2d/ps
	m  = enmap.ifft(beam2d+0j).real
	m /= m[0,0]
	norm = enmap.ifft(enmap.fft(m)*filter).real[0,0]
	filter /= norm
	return filter

def get_thumb(map, size, normalize=False):
	if normalize: map = map/map[...,0,0]
	return enmap.shift(map, (size//2, size//2), keepwcs=True)[:size,:size]

def calc_beam_transform_area(beam_2d, unit="phys"):
	"""Compute the solid angle of an l-space 2d beam in steradians"""
	area = beam_2d[0,0]/np.mean(beam_2d)
	if   unit == "phys": area *= beam_2d.pixsize()
	elif unit == "pix":  area *= 1
	else: raise ValueError("Unrecognized unit '%s' in calc_beam_transform_area" % unit)
	return area

def calc_beam_profile_area(beam_profile):
	r, b = beam_profile
	return integrate.simpson(2*np.pi*r*b,x=r)

def fit_labeled_srcs(fmap, labels, inds, extended_threshold=1.1):
	# Our normal fit is based on the center of mass. This is
	# probably a bit suboptimal for faint sources, but those will
	# be pretty bad anyway.
	pos_com = np.array(ndimage.center_of_mass(fmap, labels, inds))
	amp_com = fmap.at(pos_com.T, unit="pix")
	negative= amp_com < 0
	# We compare these amplitudes with the maxima. Normally these
	# will be very close. If they are significantly different, then
	# this is probably an extended object. To allow the description
	# of these objects as a sum of sources, it's most robust to use
	# the maximum positions and amplitudes here.
	pos_max = np.array(ndimage.maximum_position(fmap, labels, inds))
	amp_max = np.array(ndimage.maximum(fmap, labels, inds))
	pos_min = np.array(ndimage.minimum_position(fmap, labels, inds))
	amp_min = np.array(ndimage.minimum(fmap, labels, inds))
	pos_ext = pos_max.copy(); pos_ext[negative] = pos_min[negative]
	amp_ext = amp_max.copy(); amp_ext[negative] = amp_min[negative]

	pos, amp = pos_com.copy(), amp_com.copy()
	extended = np.abs(amp_ext) > np.abs(amp_com)*extended_threshold
	pos[extended] = pos_max[extended]
	amp[extended] = amp_max[extended]
	return pos, amp

def calc_model(shape, wcs, ipos, template, amp=1.0):
	amp   = np.zeros(len(ipos))+amp
	model = enmap.zeros(shape, wcs, template.dtype)
	size  = np.array(template.shape)
	dbox  = np.array([[0,0],size])-size//2
	for i, pix in enumerate(ipos):
		pix0     = utils.nint(pix)
		srcmodel = fft.shift(template, pix-pix0)*amp[i]
		enmap.insert_at(model, pix0+dbox, srcmodel, op=lambda a,b:a+b, wrap=shape[-2:])
	return model

def sim_initial_noise(div, lknee=None, alpha=-2, seed=0):
	# Simulate white noise, but temporarily switch to a fixed seed to avoid getting
	# randomness in the point source fitter output.
	if lknee is None: lknee = 3000
	rng   = np.random.RandomState(seed)
	noise = enmap.ndmap(rng.standard_normal(div.shape).astype(div.dtype), div.wcs)
	l     = div.modlmap()
	profile = (1 + ((l+0.5)/lknee)**alpha)**0.5
	profile[0,0] = 0
	#np.savetxt("profile.txt", np.array(profile.lbin()).T, fmt="%15.7e")
	noise  = enmap.ifft(enmap.fft(noise)*profile).real
	noise[div>0] *= div[div>0]**-0.5
	return noise

def amax(arr, initial=None):
	try: return np.max(arr, initial=initial)
	except TypeError:
		if initial is None: return np.max(arr)
		else: return np.max(np.concatenate([arr.reshape(-1),[initial]]))

def build_prior(amps, damps, variability=1.0, min_ivar=1e-10):
	"""Take a set of previous source amplitudes amps +- damps and turn
	them into a prior for a new fit. Variability controls how variable
	the source is assumed to be, and effectively weakens the prior."""
	n     = len(amps)
	amps[~np.isfinite(amps)] = 0
	damps[~np.isfinite(damps)] = 0
	mask  = damps > 0
	ivars = np.zeros(n)
	ivars[mask]  = 1/(damps[mask]**2 + (amps[mask]*variability)**2)
	ivars[~mask] = min_ivar
	return amps, ivars

def find_srcs(imap, idiv, beam, freq=150, apod=15, snmin=3.5, npass=2, snblock=2.5, nblock=10,
		ps_res=2000, pixwin=True, pixwin_order=0, kernel=256, dump=None, verbose=False, apod_margin=10, hack=0):
	# Apodize a bit before any fourier space operations
	apod_map = (idiv*0+1).apod(apod) * get_apod_holes(idiv,apod)
	imap = imap*apod_map
	# Deconvolve the pixel window from the beginning, so we don't have to worry about it
	if pixwin: imap = enmap.unapply_window(imap,order=pixwin_order)
	# Whiten the map
	wmap   = imap * idiv**0.5
	adiv   = idiv * apod_map**2
	beam2d = calc_2d_beam(beam, imap.shape, imap.wcs)
	beam_area = calc_beam_transform_area(beam2d)
	#print "max(imap)", np.max(imap)
	#print "median(adiv)**-0.5", np.median(adiv)**-0.5
	#print "max(wmap)", np.max(wmap), np.max(imap)/np.median(adiv)**-0.5
	# We need a point-source free map to build the noise model, and we
	# need a noise model to find the point sources. So start with a
	# dummy point source free map and then do another pass after we've
	# built a real source free map. So typically npass will be 2.
	noise  = sim_initial_noise(idiv)
	for ipass in range(npass):
		wnoise = noise * adiv**0.5
		# From now on we treat the whitened map as the real one. And assume that
		# we only need a constant covariance model. If div has lots of structure
		# on the scale of the signal we're looking for, then this could introduce
		# false detections. Empirically this hasn't been a problem, though.
		ps       = measure_noise(wnoise, apod, apod, ps_res=ps_res)
		if hack: ps = planck_hack(ps, hack)
		filter   = build_filter(ps, beam2d)
		template = get_thumb(enmap.ifft(filter*beam2d+0j).real, size=kernel, normalize=True)
		fmap     = enmap.ifft(filter*enmap.fft(wmap)).real   # filtered map
		fnoise   = enmap.ifft(filter*enmap.fft(wnoise)).real # filtered noise
		norm     = get_snmap_norm(fnoise*(apod_map==1))
		if dump:
			enmap.write_map(dump + "wnoise_%02d.fits" % ipass, wnoise)
			enmap.write_map(dump + "wmap_%02d.fits"   % ipass, wmap)
			enmap.write_map(dump + "fmap_%02d.fits"   % ipass, fmap)
			enmap.write_map(dump + "norm_%02d.fits"   % ipass, norm)
			enmap.write_map(dump + "ps_%02d.fits"     % ipass, ps)
			enmap.write_map(dump + "filter_%02d.fits" % ipass, filter)
			enmap.write_map(dump + "template_%02d.fits" % ipass, template)
		del wnoise
		result = bunch.Bunch(snmap=fmap/norm)
		fits   = bunch.Bunch(amp=[], damp=[], pix=[], npix=[])
		# We could fit all the sources in one go, but that could lead to
		# false positives from ringing around strong sources, or lead to
		# weaker sources being masked by strong ones. So we fit in blocks
		# of source strength.
		sn_lim = np.max(np.abs(fmap/norm)*(apod_map>0))
		for iblock in range(nblock):
			snmap   = fmap/norm
			if dump: enmap.write_map(dump + "snmap_%02d_%02d.fits" % (ipass, iblock), snmap)
			# Find all significant candidates, even those below our current block cutoff.
			# We do this because we will later compute a weighted average position, and we
			# want to use more than just a few pixels near the peak for that average.
			# We want both negative and positive matches, but we want to keep them separate
			# avoid averaging over positive and negative areas that are just next to each other.
			matches_pos =  snmap >= snmin
			matches_neg = -snmap >= snmin
			labels_pos, nlabel_pos = ndimage.label(matches_pos)
			labels_neg, nlabel_neg = ndimage.label(matches_neg)
			# We know that pos and neg labels have no overlap, so this should be safe
			matches= matches_pos+ matches_neg
			labels = labels_pos + labels_neg + matches_neg*nlabel_pos
			nlabel = nlabel_pos + nlabel_neg
			del matches_pos, matches_neg, labels_pos, labels_neg
			if nlabel == 0: break
			all_inds = np.arange(nlabel)
			sn       = ndimage.maximum(np.abs(snmap), labels, all_inds+1)
			# Only keep the strongest sources for now, and update sn_lim
			sn_lim   = max(snmin, min(sn_lim, np.max(sn))/snblock)
			keep     = np.where(sn >= sn_lim)[0]
			if len(keep) == 0: break
			# Measure the properties of the selected sources. This will be based on the
			# pixels that were > snmin.
			pix, amp = fit_labeled_srcs(fmap, labels, keep+1)
			damp     = norm.at(pix.T, unit="pix", order=0)
			npix     = ndimage.sum(matches, labels, keep+1)
			model    = calc_model(fmap.shape, fmap.wcs, pix, template, amp)
			# Subtract these sources from fmap in preparation for the next pass
			fmap    -= model
			fits.amp.append(amp)
			fits.damp.append(damp)
			fits.pix.append(pix)
			fits.npix.append(npix)
			if verbose:
				edges = [0,5,10,20,50,100,np.inf]
				sns   = np.concatenate(fits.amp)/np.concatenate(fits.damp)
				counts= np.histogram(sns, edges)[0]
				desc  = " ".join(["%d: %5d" % (e,c) for e,c in zip(edges[:-1],counts)])
				print("pass %d block %2d sn: %s" % (ipass+1, iblock+1, desc))
				sys.stdout.flush()
			# No point in continuing if we've already reached sn_lim < snmin. At this point
			# we're just digging into the noise.
			if sn_lim <= snmin: break
		# Construct our output catalog format
		if len(fits.amp) > 0:
			for key in fits: fits[key] = np.concatenate(fits[key])
			nsrc = len(fits.amp)
		else: nsrc = 0
		cat = np.zeros(nsrc, cat_dtype).view(np.recarray)
		if nsrc > 0:
			rms  = adiv.at(fits.pix.T, unit="pix", order=0)**-0.5
			cat.dec, cat.ra = wmap.pix2sky(fits.pix.T)
			cat.amp[:,0]  = fits.amp*rms
			cat.damp[:,0] = fits.damp*rms
			# Should not be necessary due to np.zeros above, but they appear to be uninitialized
			cat.amp[:,1:] = cat.damp[:,1:] = 0
			cat.npix = fits.npix
			# Get fluxes. 1e9 is for GHz, 1e6 is for uK
			fluxconv = utils.flux_factor(beam_area, freq*1e9)/1e6
			cat.flux  = cat.amp *fluxconv
			cat.dflux = cat.damp*fluxconv
			# Order by S/N
			cat = cat[np.argsort(cat.amp[:,0]/cat.damp[:,0])[::-1]]
			# Reject any sources that are in the apodization region
			dist_from_apod = ndimage.distance_transform_edt(apod_map>=1)
			ipix           = utils.nint(imap.sky2pix([cat.dec,cat.ra]))
			untainted = (dist_from_apod[tuple(ipix)] >= apod_margin) & (np.isfinite(rms))
			cat = cat[untainted]
		del fits
		nsrc = len(cat)
		# Compute model and residual in real units
		result.resid_snmap = fmap/norm
		beam_thumb  = get_thumb(enmap.ifft(beam2d+0j).real, size=kernel)
		beam_thumb /= np.max(beam_thumb)
		if nsrc > 0:
			pix                = imap.sky2pix([cat.dec,cat.ra]).T
			result.model       = calc_model(imap.shape, imap.wcs, pix, beam_thumb, cat.amp[:,0])
		else: result.model = imap*0
		result.resid       = imap - result.model
		result.map         = imap
		result.beam_thumb  = beam_thumb
		result.cat         = cat
		# Prepare for next iteration
		noise = result.resid
	return result

def measure_corrlen(tfun, tol=1e-2):
	"""Given a transfer function (from a beam, for example) compute its correlation
	length, which is defined at the distance in radians beyond which the correlation
	function no longer exceeds tol relative to the peak value."""
	corrfun  = enmap.ifft(tfun+0j).real
	corrfun /= corrfun[0,0]
	# Move the corrfun to the center of the area, and get distances from that point
	refpix   = np.array(tfun.shape[-2:])//2
	refpos   = corrfun.pix2sky(refpix)
	corrfun  = enmap.shift(corrfun, refpix, keepwcs=True)
	r        = corrfun.modrmap(ref=refpos)
	# Find the highest radius where we're above the tolerance
	corrlen  = np.max(r[np.abs(corrfun)>tol])
	return corrlen

def group_independent(pos, corrlen):
	pos   = utils.rewind(pos)
	n     = len(pos)
	# Add angle wrapped version of the positions
	wpos1 = pos.copy(); wpos1[:,1] -= 2*np.pi
	wpos2 = pos.copy(); wpos2[:,1] += 2*np.pi
	wpos  = np.concatenate([pos,wpos1,wpos2],0)
	del wpos1, wpos2
	# Apply cos dec scaling to make KDTree distance computation approximate
	# angular distances
	pos[:,1]  *= np.cos(pos[:,0])
	wpos[:,1] *= np.cos(wpos[:,0])
	# Find all points that are correlated with each point
	tree  = spatial.cKDTree(pos)
	wtree = spatial.cKDTree(wpos)
	corr_groups= tree.query_ball_tree(wtree, corrlen)
	# Normalize wrapped points
	corr_groups= [set([i%n for i in g]) for g in corr_groups]
	# Split into groups with the property that all the points in each group are indendent.
	# This algorithm isn't that efficient, but the number of sources won't be *that* big
	indep_groups = []
	remainder    = set(range(n))
	while len(remainder) > 0:
		group = []
		candidates = remainder.copy()
		while len(candidates) > 0:
			# Get an element from the candidates
			elem = candidates.pop()
			# Remove all points correlated with elem from candidates
			candidates -= corr_groups[elem]
			group.append(elem)
		remainder -= set(group)
		indep_groups.append(sorted(group))
	# Turn all the sets into lists in groups, so our output is a bit cleaner
	corr_groups = [list(g) for g in corr_groups]
	return indep_groups, corr_groups

# FIXME: Need to import the complicated pixel window stuff from jointmap to
# be able to handle planck. This is duplicating jointmap quite a lot...
# Is there a nice way to merge them? There are two main differences:
# 1. diff-based vs. tot-based. Dory does not need diff-maps or background
#    spectra, but instead needs to iterate to get the noise model right.
# 2. arguments vs. data structures: jointmap has a mapdata structure which
#    is initialized from a config file. Dory wants to be simpler. But it's
#    starting to be quite a few things that need to be passed in:
#    map, ivar, beam, freq and pixwin.

class FitError(Exception): pass

def fit_src_amps(imap, idiv, src_pos, beam, prior=None,
		apod=15, npass=2, indep_tol=1e-4, ps_res=500, pixwin=True, pixwin_order=0, beam_tol=1e-4,
		dump=None, verbose=False, apod_margin=10, hack=0, region=0, lknee=None, maxcorrlen=3*utils.arcmin, lmin=0):
	# Get the (fractional) pixel positions of each source
	t1 = time.time()
	src_pix  = imap.sky2pix(src_pos.T).T
	# We will only fit sources that are inside our area, and which are not contaminated
	# by apodization.
	margin   = apod+apod_margin
	fit_inds = np.where(np.all(src_pix >= margin, 1) & np.all(src_pix < np.array(imap.shape[-2:])-margin, 1))[0]
	src_pos, src_pix = src_pos[fit_inds], src_pix[fit_inds]
	nsrc     = len(src_pos)
	if len(src_pos) == 0:
		return fit_inds, np.zeros([0]), np.zeros([0,0]), np.zeros(nsrc)
	# Apodize a bit before any fourier space operations
	apod_map = (idiv*0+1).apod(apod) * get_apod_holes(idiv,apod)
	imap     = imap*apod_map
	if dump:
		enmap.write_map(dump + "imap.fits", imap)
		enmap.write_map(dump + "idiv.fits", idiv)
	# We should either handle the polarization looping inside this function,
	# or possibly always return something for all the input sources. As it is,
	# we can have any logic here that would select different sources for different
	# components.. Well, we could fix the calling function I guess..
	# If the region isn't hit at all we can't build a noise model, nor is there
	# anything to measure, so just bail out.
	if np.sum(idiv*apod_map**2) == 0:
		return fit_inds, np.zeros(nsrc), np.zeros([nsrc,nsrc]), np.zeros(nsrc)
	# Deconvolve the pixel window from the beginning, so we don't have to worry about it
	if pixwin: imap = enmap.unapply_window(imap, order=pixwin_order)
	beam_prof = get_beam_profile(beam)
	# Find the distance at which point we have fallen to beam_tol
	brad      = get_beam_rad(beam_prof, beam_tol)
	# Build a beam for each source. This shouldn't be too expensive, as each only
	# will cover the pixels necessary.
	posmap    = imap.posmap()
	pboxes    = enmap.neighborhood_pixboxes(imap.shape, imap.wcs, src_pos, brad)
	Bs        = []
	for sid, pbox in enumerate(pboxes):
		pos  = posmap.extract_pixbox(pbox)
		r    = utils.angdist(pos[::-1], src_pos[sid,::-1,None,None])
		bpix = (r - beam_prof[0,0])/(beam_prof[0,1]-beam_prof[0,0])
		bval = enmap.samewcs(utils.interpol(beam_prof[1], bpix[None], border="constant", order=1), pos)
		Bs.append(bval)
		#enmap.write_map("test_Bs_%02d.fits" % (sid), Bs[-1])
	# We only need these for the matched filter correlation length calculation later
	beam2d    = calc_2d_beam(beam, imap.shape, imap.wcs)
	beam2d   /= np.mean(beam2d) # normalize so that it corresponds to a profile starting at 1
	# The enmap symmetric fourier space unit convention is not good for convolutions, so
	# switch to the fft one.
	def map_fft(m):  return enmap.fft(m, normalize=False)
	def map_ifft(m): return enmap.ifft(m, normalize=False).real/m.npix
	# We will use a constant correlation model when doing these fits. This is
	# slightly different from the whiten+const-cov model used in find_srcs, since
	# it doesn't implicitly assume that the point source profile itself is modulated
	# by the hitcounts. We can afford that here because we know where the sources are.
	H      = idiv**0.5 * apod_map
	noise  = sim_initial_noise(idiv, lknee=lknee)
	t2 = time.time()
	if verbose: print("%8.2f Prepare" % (t2-t1))
	for ipass in range(npass):
		# Build the noise model based on the current noise map
		C          = measure_noise(H*noise, apod, apod, ps_res=ps_res)
		if hack: C = planck_hack(C, hack)
		if np.sum(C) == 0: raise FitError("No data in region")
		iC         = 1/C
		# Optionally zero out inverse variance for some scales we want to ignore
		if lmin: iC *= iC.modlmap()<lmin
		#np.savetxt("C.txt", np.array(C.lbin()).T, fmt="%15.7e")
		#np.savetxt("iC.txt", np.array(iC.lbin()).T, fmt="%15.7e")
		#np.savetxt("BiC.txt", np.array((beam2d*iC).lbin()).T, fmt="%15.7e")
		#enmap.write_map("test_iC.fits", iC)
		rhs  = np.zeros([nsrc])
		icov = np.zeros([nsrc,nsrc])
		# We can now build our rhs
		t1 = time.time()
		Nd = H*map_ifft(iC*map_fft(H*imap))
		#enmap.write_map("Nd.fits", Nd)
		#enmap.write_map("BNd.fits", enmap.ifft(enmap.fft(Nd)*beam2d).real)
		for sid in range(nsrc):
			rhs[sid] = np.sum(Nd.extract_pixbox(pboxes[sid])*Bs[sid])
		t2 = time.time()
		if verbose: print("%8.2f Build rhs pass %d/%d" % (t2-t1, ipass+1, npass))
		# Build the icov. I used to split over indep_groups, and then loop over
		# each nearby source for each member, but this is flawed - it can't know
		# which member in the current group is responsible for the response some
		# source in the neighborhood sees.
		#
		# Instead we will use indep groups to efficiently compute NB for each source,
		# and then loop over each source's neighborhood
		corrlen  = measure_corrlen(beam2d**2*iC, indep_tol)
		corrlen  = np.minimum(corrlen, maxcorrlen)
		if verbose: print("corrlen", corrlen/utils.degree)
		cboxes   = enmap.neighborhood_pixboxes(imap.shape, imap.wcs, src_pos, corrlen)
		with bench.mark("make groups"):
			# We don't want any part of another source's matched filter inside the
			# region we're going to extract. 2**0.5 for diagonal, which is the worst
			# case, and 2 because each source contributes its radius. Could avoid
			# 2**0.5 with extra mask.
			indep_groups, corr_groups = group_independent(src_pos, corrlen*2*2**0.5)
		NBs = [None for i in range(nsrc)]
		t1 = time.time()
		for gi, igroup in enumerate(indep_groups):
			# Evaluate the covariance around every source in igroup in parallel
			NB = imap*0
			for sid in igroup:
				NB.insert(Bs[sid], op=np.add)
			NB = H*map_ifft(iC*map_fft(H*NB))
			for sid in igroup:
				NBs[sid] = NB.extract_pixbox(cboxes[sid])
				#enmap.write_map("test_NBs_%02d_%02d.fits" % (ipass, sid), NBs[sid])
		t2 = time.time()
		if verbose: print("%8.2f Built NBs pass %d/%d" % (t2-t1, ipass+1, npass))
		t1 = time.time()
		#js = [1,54,55,57,71]
		for sid in range(nsrc):
			#if sid in js:
			#	enmap.write_map("test_NBs_%02d_%02d.fits" % (ipass, sid), NBs[sid])
			for sid2 in corr_groups[sid]:
				overlap = NBs[sid].extract(Bs[sid2].shape, Bs[sid2].wcs)
				#if sid in js and sid2 in js:
				#	enmap.write_map("test_overlap_%02d_%02d_%02d.fits" % (ipass, sid, sid2), overlap)
				icov[sid,sid2] = np.sum(overlap*Bs[sid2])
		t2 = time.time()
		if verbose: print("%8.2f Built icov pass %d/%d" % (t2-t1, ipass+1, npass))
		#np.save("test_rhs1_%02d.npy" % ipass, rhs)
		#np.save("test_icov1_%02d.npy" % ipass, icov)
		# Apply any prior
		if prior is not None:
			prior_amp, prior_ivar = prior
			rhs  += prior_ivar[fit_inds]*prior_amp[fit_inds]
			icov += np.diag(prior_ivar[fit_inds])
		else:
			# Pseudo-prior to avoid degenerate equation system
			ref   = np.max(icov)*1e-6
			if ref <= 0: ref = 1
			icov += np.diag(np.full(len(rhs),ref))
		#np.save("test_rhs2_%02d.npy" % ipass, rhs)
		#np.save("test_icov2_%02d.npy" % ipass, icov)
		# Our equation system can be a bit asymmetrical. This is expected at some level
		# due to the beam and correlation cutoffs, though I haven't confirmed that that's
		# really what's going on here. For now we symmetrize and hope for the best.
		icov  = 0.5*(icov+icov.T)
		amp   = np.linalg.solve(icov, rhs)
		#np.save("test_rhs3_%02d.npy" % ipass, rhs)
		#np.save("test_icov3_%02d.npy" % ipass, icov)
		#np.save("test_amp3_%02d.npy" % ipass, amp)
		#np.save("test_pos.npy", src_pos)
		#print("amp %d" % ipass)
		#print(np.sort(amp))
		damp  = np.diag(icov)**-0.5
		if prior is None:
			# Give pseudo-prior-dominated entries infinite uncertainty
			damp[damp >= ref**-0.5/2] = np.inf
		# Subtract this from the map to get a better noise estimate
		model = imap*0
		for sid in range(nsrc):
			model.insert(Bs[sid]*amp[sid], op=np.add)
		local_amps = model.at(src_pix.T, unit="pix", order=1)
		noise = imap - model
		t2 = time.time()
		if verbose: print("%8.2f Solve pass %d/%d" % (t2-t1, ipass+1, npass))
		#enmap.write_map("test_map_%02d_%d.fits" % (region, ipass), imap)
		#enmap.write_map("test_model_%02d_%d.fits" % (region, ipass), model)
		#enmap.write_map("test_resid_%02d_%d.fits" % (region, ipass), noise)
	# Prune a bit further at the edge, since an artifact associated with a source just at the
	# edge might have been included while the source itself was excluded, leading to the artifact
	# trying to absorb all the source power.
	margin += apod_margin
	good    = np.all(src_pix >= margin, 1) & np.all(src_pix < np.array(imap.shape[-2:])-margin, 1)
	fit_inds, amp, icov, local_amps = fit_inds[good], amp[good], icov[good][:,good], local_amps[good]
	# This function doesn't build a full catalog object. It just returns
	# the amplitudes and their inverse covariance.
	return fit_inds, amp, icov, local_amps

def prune_contained(icat, beam, rmax=5*utils.arcmin, beam_exp=2, merge="primary", verbose=False):
	"""Prune catalog icat by merging sources that are inside brigther ones. We
	define a source as being inside another one when its peak amplitude is lower
	than the value of the other's beam at that location. beam_exp can be used to scale
	the beam radially. beam_exp=1 corresponds to the plain beam, beam_exp==2 is close to
	that of a matched filter. Higher values can be used for ad-hoc beam broadening like
	for approximating the day-time beam, or just to be extra conservative."""
	# Get the real-space beam profile. The matched filter goes as the fourier-squared beam,
	# so square the beam first.
	beam_profile     = get_beam_profile(beam**beam_exp, nsamp=1001, tol=1e-5, rmax=rmax)
	beam_profile[1] /= beam_profile[1,0]
	dr = beam_profile[0,1]
	def beval(dist): return ndimage.map_coordinates(beam_profile[1], dist[None]/dr, order=1)
	# First sort the catalog from brightest to faintest
	SN     = np.abs(icat.amp[:,0])
	order  = np.argsort(SN)[::-1]
	icat   = icat[order]
	nsrc   = len(icat)
	# Then loop through our groups
	pos    = utils.ang2rect([icat.ra, icat.dec]).T
	amps   = icat.amp[:,0]
	tree   = spatial.cKDTree(pos)
	groups = tree.query_ball_tree(tree, rmax)
	done   = np.zeros(nsrc, bool)
	ocat   = []
	nmerged = 0
	for gi, group in enumerate(groups):
		group = np.sort(np.array(group))
		group = group[~done[group]]
		if len(group) == 0: continue
		# Get distance from the brightest group member to
		# all the other members
		primary, others = group[0], group[1:]
		dists   = utils.vec_angdist(pos[primary],pos[others],-1)
		# and use that to get the beam value at their location
		vals    = beval(dists)*amps[primary]
		# group with primary if this value exceeds their own amplitude
		inside  = others[np.abs(vals) > np.abs(amps[others])]
		ogroup  = np.concatenate([[primary],inside])
		#print("ogroup", ogroup)
		# Merge this group into a single representative source
		if merge == "primary":
			osrc = icat[primary]
		elif merge == "mean":
			gcat    = icat[ogroup]
			weight  = (gcat.amp[:,0]/gcat.damp[:,0])**2
			# Why is this so cumbersome?
			osrc    = np.array(0, icat.dtype)
			for field in gcat.dtype.fields.keys():
				osrc[field] = np.sum(gcat[field]*weight)/np.sum(weight)
		else: raise ValueError("Unknown merge mode '%s'" % (merge))
		done[ogroup] = True
		ocat.append(osrc)
		nmerged += len(inside)
		if verbose and gi % 100 == 0:
			sys.stderr.write("\r%6d/%d %6d %6d" % (gi, nsrc, len(ocat), nmerged))
	if verbose: sys.stderr.write("\n")
	ocat = np.array(ocat).view(np.recarray)
	return ocat

def prune_artifacts(result):
	# Given a result struct from find_srcs, detect artifacts and remove them both from
	# the catalog and the model and residual maps (but not from the snmap ones for now)
	result = result.copy()
	owners, artifacts = find_source_artifacts(result.cat)
	if len(artifacts) == 0: return result
	all_arts = np.concatenate(artifacts)
	good     = np.full(len(result.cat), True, bool)
	good[all_arts] = False
	result.cat = result.cat[good]
	# Build new model
	pix          = result.map.sky2pix([result.cat.dec, result.cat.ra]).T
	result.model = calc_model(result.map.shape, result.map.wcs, pix, result.beam_thumb, result.cat.amp[:,0])
	result.resid = result.map - result.model
	return result

def prune_near_bright(cat, lim_bright=100, rlim=2*utils.arcmin):
	snr    = np.abs(cat.flux[:,0]/cat.dflux[:,0])
	bright = snr > lim_bright
	cat.ra = utils.rewind(cat.ra, 0)
	pos    = np.array([cat.ra*np.cos(cat.dec),cat.dec]).T
	tree_all    = spatial.cKDTree(pos)
	tree_bright = spatial.cKDTree(pos[bright])
	groups      = tree_bright.query_ball_tree(tree_all, rlim)
	rejected    = np.zeros(len(cat), bool)
	for gi, group in enumerate(groups):
		sns  = snr[group]
		bad  = np.full(len(group), True, bool)
		bad[np.argmax(sns)] = False
		rejected[group] |= bad
	return cat[~rejected]

def write_catalog(ofile, cat):
	if ofile.endswith(".fits"): write_catalog_fits(ofile, cat)
	else: write_catalog_txt (ofile, cat)

def read_catalog(ifile):
	if ifile.endswith(".fits"): return read_catalog_fits(ifile)
	else: return read_catalog_txt(ifile)

def write_catalog_txt(ofile, cat):
	np.savetxt(ofile, np.array([
		cat.ra/utils.degree,
		cat.dec/utils.degree,
		cat.amp[:,0]/cat.damp[:,0],
		cat.amp[:,0]/1e3, cat.damp[:,0]/1e3,
		cat.amp[:,1]/1e3, cat.damp[:,1]/1e3,
		cat.amp[:,2]/1e3, cat.damp[:,2]/1e3,
		cat.flux[:,0]*1e3, cat.dflux[:,0]*1e3,
		cat.flux[:,1]*1e3, cat.dflux[:,1]*1e3,
		cat.flux[:,2]*1e3, cat.dflux[:,2]*1e3,
		cat.npix, cat.status,
	]).T, fmt="%11.6f %11.6f %8.3f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %5.0f %2.0f",
	header = "ra dec SNR Tamp dTamp Qamp dQamp Uamp dUamp Tflux dTflux Qflux dQflux Uflux dUflux npix status")

def write_catalog_fits(ofile, cat):
	from astropy.io import fits
	ocat = cat.copy()
	for field in ["ra","dec"]:     ocat[field] /= utils.degree # angles in degrees
	for field in ["amp","damp"]:   ocat[field] /= 1e3          # amplitudes in mK
	for field in ["flux","dflux"]: ocat[field] *= 1e3          # fluxes in mJy
	hdu = fits.hdu.table.BinTableHDU(ocat)
	hdu.writeto(ofile, overwrite=True)

def read_catalog_txt(ifile):
	data = np.loadtxt(ifile, ndmin=2).T
	cat  = np.zeros(data.shape[1], cat_dtype).view(np.recarray)
	cat.ra, cat.dec = data[0:2]*utils.degree
	cat.amp  = data[3:9:2].T*1e3
	cat.damp = data[4:9:2].T*1e3
	cat.flux = data[9:15:2].T/1e3
	cat.dflux= data[10:15:2].T/1e3
	cat.npix = data[15]
	cat.status = data[16]
	return cat

def read_catalog_fits(fname):
	from astropy.io import fits
	hdu = fits.open(fname)[1]
	cat = np.asarray(hdu.data).view(np.recarray)
	for field in ["ra","dec"]:     cat[field] *= utils.degree # deg -> rad
	for field in ["amp","damp"]:   cat[field] *= 1e3          # mK  -> uK
	for field in ["flux","dflux"]: cat[field] /= 1e3          # mJy -> Jy
	return cat

def allgather_catalog(cat, comm):
	# This is hacky. It only works if all the columns of cat are floats. Which they are.
	# But it's still ugly. I wish mpi4py supported recarrays.
	def to_2d(arr): return arr if arr.ndim == 2 else arr[:,None]
	fields = [to_2d(cat[key]) for key in cat.dtype.fields]
	inds   = utils.cumsum([field.shape[1] for field in fields], endpoint=True)
	stacked= np.concatenate(fields,1)
	total  = utils.allgatherv(stacked, comm, axis=0)
	res    = np.zeros(len(total), dtype=cat.dtype).view(np.recarray)
	for i, key in enumerate(cat.dtype.fields):
		res[key] = total[:,inds[i]:inds[i+1]].reshape(res[key].shape)
	return res

def find_source_artifacts(cat, vlim=0.005, maxrad=80*utils.arcmin, jumprad=7*utils.arcmin,
		gmax=1000, maxit=100, core_lim=0.05, core_rad=2*utils.arcmin):
	"""Find artifacts in the source database that are sourced by point sources. This consists
	of two parts: Core artifacts, which are very close to the source that caused them, and
	are due to beam or position inaccuracies in the fit; and X artifacts, which are due to
	x-bleeding during mapmaking. These are fit in the same function because the boundary between
	these two classes can be vague, and skipping the core artifacts can make us miss the beginning
	of the chain of sources the x-finding algorithm uses.

	Returns owners[n] arts[n][nart], where owners is an array of the indices of the sources
	that sourced the artifacts, and args is a list of arrays of the indices of the artifact
	sources.
	
	The numbers used here are quite ACT-specific, but then so are X-artifacts.
	"""
	# Find X artifacts in the map by finding sources that are connected to a bright
	# source by a series of jumps no longer than jumprad.
	if len(cat) == 0: return np.zeros([0],int), []
	sn     = cat.amp[:,0]/cat.damp[:,0]
	pos    = np.array([cat.ra,cat.dec]).T
	cpos   = pos.copy(); cpos[:,0] *= np.cos(pos[:,1])
	tree   = spatial.cKDTree(cpos)
	strong = np.where(sn > 1/core_lim)[0]
	if len(strong) == 0: return np.zeros([0],int), []
	# Subtract stripes
	tree_strong = spatial.cKDTree(cpos[strong])
	groups      = tree_strong.query_ball_tree(tree, maxrad)
	# Sort groups by S/N, so artifacts that are themselves strong only get
	# counted once
	order       = np.argsort(sn[strong])[::-1]
	done        = set()
	owners, artifacts = [], []
	for gi in order:
		si    = strong[gi]
		group = np.array(groups[gi])
		if si in done: continue
		# First find the core artifacts
		center_dist = utils.angdist(pos[group], pos[si,None], axis=1)
		core_mask   = (sn[group] < sn[si]*core_lim) & (center_dist < core_rad)
		# We want to measure distance from the main source and the core group to begin with
		tagged = set([si]) | set(group[core_mask])
		# For the actual X we restrict ourselves to weaker sources, so we don't
		# throw away nearby real sources
		group = group[sn[group] < sn[si]*vlim]
		# If there are too many nearby sources, then something weird is going on in
		# this area, and that weird stuff probably isn't an X artifact
		if len(group) > 0 and len(group) < gmax:
			candidates = set(group)
			for it in range(maxit):
				rest    = candidates-tagged
				tarr, rarr = [np.array(list(s)) for s in [tagged, rest]]
				dists   = utils.angdist(pos[rarr,None], pos[None,tarr], axis=2)
				mindist = np.min(dists,1)
				matches = rarr[mindist < jumprad]
				if len(matches) == 0: break
				tagged.update(matches)
		# Remove the original strong source again, it will be listed separately
		tagged.remove(si)
		if len(tagged) == 0: continue
		done.update(tagged)
		owners.append(si)
		artifacts.append(list(tagged))
	return owners, artifacts

def merge_duplicates(cat, rlim=1*utils.arcmin, alim=0.25, uncertainty="min"):
	"""Given a point source catalog which might contain duplicates, detect these duplicates
	and merge them to produce a single catalog with no duplicates. Sources are considered
	duplicates if they are within rlim of each other. Merging uses averaging if the amplitudes
	differ by less than alim fractionally. Otherwise the strongest one is used. This is to prevent
	a strong source from being averaged with its own artifacts. rlim should be adjusted
	to fit the exerpiment beam. The default is appropriate for ACT."""
	if len(cat) == 0: return cat
	# Normalize positions first. This could miss some mergers on the edge.
	cat    = cat.copy()
	cat.ra = utils.rewind(cat.ra, 0)
	pos    = np.array([cat.ra*np.cos(cat.dec),cat.dec]).T
	tree   = spatial.cKDTree(pos)
	groups = tree.query_ball_tree(tree, rlim)
	done   = np.zeros(len(cat),bool)
	ocat   = []
	for gi, group in enumerate(groups):
		# Remove everything that's done
		group = np.array(group)
		group = group[~done[group]]
		if len(group) == 0: continue
		# Nothing to do for groups with only one member
		if len(group) == 1:
			done[group[0]] = True
			ocat.append(cat[group[0]])
		else:
			amps  = cat.amp[group,0]
			good  = np.where(np.abs(amps) >= np.max(np.abs(amps))*(1-alim))[0]
			# Nans could lead to us disqualifying all of them
			if len(good) == 0: continue
			gcat  = cat[group[good]]
			entry = np.zeros([], cat.dtype)
			def wmean(v, w):
				if np.any(~np.isfinite(v.T)) or np.any(~np.isfinite(w)):
					print(v, w, np.isfinite(v), np.isfinite(w))
				rhs = np.sum(v.T*w.T,-1)
				div = np.sum(w.T,-1)
				div = np.maximum(div, 1e-30)
				return (rhs/div).T
			def nonan(a): return np.where(np.isfinite(a),a,0)
			for key in cat.dtype.fields:
				# Weighted mean in case one is more uncertain for some reason
				if gcat[key].ndim == 2:
					entry[key] = wmean(nonan(gcat[key]), nonan(gcat["damp"]**-2))
				else:
					entry[key] = wmean(nonan(gcat[key]), nonan(gcat["damp"][:,0]**-2))
			# Set min uncertainty of inputs as the effective one. Could have
			# also used np.mean(gcat.damp**-2)**-0.5, but I trust the least uncertain
			# one more.
			if uncertainty == "min":
				entry["damp"] = np.min(gcat["damp"],0)
			else:
				entry["damp"] = np.sum(nonan(gcat["damp"]**-2))**-0.5
			# Handle the integer fields
			entry["status"] = np.median(gcat["status"])
			ocat.append(entry)
			done[group] = True
	ocat = np.array(ocat).view(np.recarray)
	return ocat

def remove_duplicates_chain(cat, rlim=1*utils.arcmin):
	"""Given a point source catalog which might contain duplicates, detect and remove
	these duplicates, returning a deduplicated catalog. Unlike merge_duplicates, this continues
	in a chain, so that neighbors of a duplicate also are considered duplicates. No averaging
	is done - the strongest source in each group is used.
	rlim should be adjusted to fit the exerpiment beam. The default is appropriate for ACT."""
	if len(cat) == 0: return cat
	# Sort the catalog by S/N
	sn     = np.abs(cat.amp[:,0]/cat.damp[:,0])
	cat    = cat[np.argsort(sn)[::-1]].copy()
	pos    = utils.ang2rect([cat.ra, cat.dec]).T
	tree   = spatial.cKDTree(pos)
	owner  = np.full(len(cat),-1,int)
	groups = tree.query_ball_tree(tree, rlim)
	for gi, group in enumerate(groups):
		# Set ourself as owner we we don't have one. Otherwise get the id of our
		# owner's owner
		if owner[gi] < 0: owner[gi] = gi
		else: gi = owner[gi]
		# Mark all members as owned
		for id in group: owner[id] = gi
	# Remove everything that's owned by someone else
	ocat = cat[owner==np.arange(len(cat))]
	return ocat

def eval_flux_at_srcs(cat, beam_profile, tol=1e-5, verbose=False):
	"""Get the contribution from all sources at the location of each source"""
	flux  = np.zeros(len(cat))
	r, br = beam_profile
	rmax  = r[br>tol][-1]
	pos   = utils.ang2rect([cat.ra, cat.dec]).T
	tree  = spatial.cKDTree(pos)
	groups= tree.query_ball_tree(tree, rmax)
	for gi, group in enumerate(groups):
		if verbose: sys.stderr.write("\r%5d/%d %3d%%" % (gi+1, len(cat), 100.0*(gi+1)/len(cat)))
		# Get our distance to everybody in our group. The tree
		# should already know this, but I don't think we can get at it
		dists = utils.vec_angdist(pos[group], pos[None,gi], axis=1)
		vals  = np.interp(dists, r, br)*cat.flux[gi,0]
		flux[group] += vals
	if verbose: sys.stderr.write("\n")
	return flux

def build_merge_weight(shape, dtype=np.float64):
	ny, nx = shape[-2:]
	cy, cx = (np.array(shape[-2:])-1)/2.0
	yoff   = np.abs(np.arange(ny)-cy)
	xoff   = np.abs(np.arange(nx)-cx)
	wy     = (1-2*yoff/ny).astype(dtype)
	wx     = (1-2*xoff/nx).astype(dtype)
	weights = wy[:,None]*wx[None,:]
	return weights

def merge_maps_onto(maplist, shape, wcs, comm, root=0, crop=0, dtype=None):
	if dtype is None: dtype = comm.bcast(maplist[0].dtype.char)
	dtype = utils.fix_dtype_mpi4py(dtype)
	pre = tuple(shape[:-2])
	# First crop the maps if necessary
	if crop: maplist = [map[...,crop:-crop,crop:-crop] for map in maplist]
	if comm.rank == root:
		omap = enmap.zeros(shape, wcs, dtype)
		odiv = enmap.zeros(shape, wcs, dtype)
		for ri in range(comm.size):
			if comm.rank == ri:
				# Handle self-to-self
				for imap in maplist:
					idiv = enmap.samewcs(build_merge_weight(imap.shape, dtype),imap)
					enmap.insert(omap, imap*idiv, op=np.add)
					enmap.insert(odiv, idiv,      op=np.add)
			else:
				# Then handle those of other tasks
				nmap = comm.recv(source=ri)
				for i in range(nmap):
					pbox = np.array(comm.recv(source=ri)).reshape(2,2)
					imap = np.zeros(pre + tuple(pbox[1]-pbox[0]), dtype)
					idiv = build_merge_weight(imap.shape, dtype)
					comm.Recv(imap,   source=ri)
					enmap.insert_at(omap, pbox, imap*idiv, op=np.add)
					enmap.insert_at(odiv, pbox, idiv,      op=np.add)
		with utils.nowarn():
			omap /= odiv
			omap  = np.nan_to_num(omap, copy=False)
		return omap
	else:
		# Send our data to root
		nmap = len(maplist)
		comm.send(nmap, dest=root)
		for i in range(nmap):
			imap = maplist[i]
			pbox = enmap.pixbox_of(wcs, imap.shape, imap.wcs)
			comm.send(list(pbox.reshape(-1)), dest=root)
			comm.Send(np.ascontiguousarray(imap), dest=root)
		return None

def get_beam_profile(beam, nsamp=10001, rmax=0, tol=1e-7):
	# First do a low-res run to find rmax
	if not rmax:
		r0   = np.linspace(0, np.pi, nsamp)
		br0  = utils.beam_transform_to_profile(beam, r0, normalize=True)
		above= np.where(br0 > tol)[0]
		imax = min(len(r0)-1,above[-1]+1 if len(above)>0 else len(br0))
		rmax = r0[imax]
	# Then get the actual profile
	r    = np.linspace(0, rmax, nsamp)
	br   = utils.beam_transform_to_profile(beam, r, normalize=True)
	B    = np.array([r,br])
	return B

def get_beam_rad(beam, lim=1e-4):
	above = np.where(beam[1]>lim)[0]
	i     = above[-1] if len(above) > 0 else -1
	return beam[0,i]

def split_sources(icat, nimage=4, dist=0.5*utils.arcmin, minflux=0.1):
	bright = icat.flux[:,0] > minflux
	print(bright.shape, icat.shape)
	if np.sum(bright) == 0: return icat
	cat_faint  = icat[~bright]
	cat_bright = icat[bright]
	ocat = [cat_bright, cat_faint]
	for i in range(nimage):
		ang  = 2*np.pi*i/nimage
		ddec = np.sin(ang)*dist
		dra  = np.cos(ang)*dist/np.cos(cat_bright.dec)
		wcat = cat_bright.copy()
		wcat.dec += ddec
		wcat.ra  += dra
		# The extra images should have zero amplitude prior, so they are only excited if needed
		wcat.flux = 0
		wcat.amp  = 0
		ocat.append(wcat)
	ocat = np.concatenate(ocat).view(np.recarray)
	return ocat
