import numpy as np
from scipy import ndimage, spatial
from enlib import enmap, utils, bunch, mpi, fft

cat_dtype = [("x","f"),("y","f"),("ra","f"),("dec","f"),("amp","f"),("damp","f"),("npix","f")]

def get_beam(fname):
	try:
		sigma = float(fname)*utils.arcmin*utils.fwhm
		l     = np.arange(40e3)
		beam  = np.exp(-0.5*(l*sigma)**2)
	except ValueError:
		beam = np.loadtxt(fname, usecols=(1,))
	return beam

def get_regions(fname, shape, wcs):
	# Set up our regions. No significant automation for now.
	if fname:
		# Region file has format ra1 ra2 dec1 dec2. Make it
		# [:,{from,to},{dec,ra}] so it's compatible with enmap bounding boxes
		regions  = np.loadtxt(fname)[:,:4]
		regions  = np.transpose(regions.reshape(-1,2,2),(0,2,1))[:,:,::-1]
		regions *= utils.degree
		# And turn them into pixel bounding boxes
		regions = np.array([enmap.skybox2pixbox(shape, wcs, box) for box in regions])
		regions = np.round(regions).astype(int)
	else:
		regions = np.array([[0,0],shape[-2:]])[None]
	return regions

def pad_region(region, pad):
	region = np.array(region)
	region[...,0,:] -= pad
	region[...,1,:] += pad
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
	ps     = smooth_ps_gauss(ps, ps_res)
	#enmap.write_map("ps2.fits", ps*0+np.fft.fftshift(ps))
	return ps

def build_filter(ps, beam):
	# Build our matched filter, assumping beam-shaped point sources
	lmap   = ps.modlmap()
	beam2d = np.interp(lmap, np.arange(len(beam)), beam)
	filter = beam2d/ps
	# Construct the 
	m  = enmap.ifft(beam2d+0j).real
	m /= m[0,0]
	norm = enmap.ifft(enmap.fft(m)*filter).real[0,0]
	filter /= norm
	return filter, beam2d

def get_thumb(map, size):
	return enmap.shift(map, (size//2, size//2))[:size,:size]

def get_template(filter, beam2d, size):
	# Build the real-space template representing the
	# response of the filter to a unit-amplitude point source
	template  = enmap.ifft(filter*beam2d+0j).real
	template /= np.max(template)
	template  = get_thumb(template, size=size)
	return template

def fit_srcs(fmap, labels, inds, extended_threshold=1.1):
	# Our normal fit is based on the center of mass. This is
	# probably a bit suboptimal for faint sources, but those will
	# be pretty bad anyway.
	pos_com = np.array(ndimage.center_of_mass(fmap, labels, inds))
	amp_com = fmap.at(pos_com.T, unit="pix")
	# We compare these amplitudes with the maxima. Normally these
	# will be very close. If they are significantly different, then
	# this is probably an extended object. To allow the description
	# of these objects as a sum of sources, it's most robust to use
	# the maximum positions and amplitudes here.
	pos_max = np.array(ndimage.maximum_position(fmap, labels, inds))
	amp_max = np.array(ndimage.maximum(fmap, labels, inds))
	pos, amp = pos_com.copy(), amp_com.copy()
	extended = amp_max > amp_com*extended_threshold
	pos[extended] = pos_max[extended]
	amp[extended] = amp_max[extended]
	return pos, amp

def calc_model(shape, wcs, ipos, amp, template):
	model = enmap.zeros(shape, wcs, template.dtype)
	size  = np.array(template.shape)
	dbox  = np.array([[0,0],size])-size//2
	for i, pix in enumerate(ipos):
		pix0     = utils.nint(pix)
		srcmodel = fft.shift(template, pix-pix0)*amp[i]
		enmap.insert_at(model, pix0+dbox, srcmodel, op=lambda a,b:a+b, wrap=shape[-2:])
	return model

def sim_initial_noise(div, lknee=3000, alpha=-2):
	# Simulate white noise
	noise = enmap.rand_gauss(div.shape, div.wcs, div.dtype)
	l     = div.modlmap()
	profile = 1 + ((l+0.5)/lknee)**alpha
	profile[0,0] = 0
	noise  = enmap.ifft(enmap.fft(noise)*profile).real
	noise[div>0] *= div[div>0]**-0.5
	return noise

def amax(arr, initial=None):
	try: return np.max(arr, initial=initial)
	except TypeError:
		if initial is None: return np.max(arr)
		else: return np.max(np.concatenate([arr.reshape(-1),[initial]]))

def find_srcs(imap, idiv, beam, apod=15, snmin=3.5, npass=2, snblock=5, nblock=10,
		ps_res=2000, pixwin=True, kernel=256, dump=None, verbose=False, apod_margin=10):
	# Apodize a bit before any fourier space operations
	apod_map = (idiv*0+1).apod(apod) * get_apod_holes(idiv,apod)
	imap = imap*apod_map
	# Deconvolve the pixel window from the beginning, so we don't have to worry about it
	if pixwin: imap = enmap.apply_window(imap,-1)
	# Whiten the map
	wmap   = imap * idiv**0.5
	adiv   = idiv * apod_map**2
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
		ps             = measure_noise(wnoise, apod, apod, ps_res=ps_res)
		filter, beam2d = build_filter(ps, beam)
		template       = get_template(filter, beam2d, size=kernel)
		fmap           = enmap.ifft(filter*enmap.fft(wmap)).real
		fnoise         = enmap.ifft(filter*enmap.fft(wnoise)).real
		norm           = get_snmap_norm(fnoise*(apod_map==1))
		del wnoise
		if dump:
			enmap.write_map(dump + "wnoise_%02d.fits" % ipass, wnoise)
			enmap.write_map(dump + "wmap_%02d.fits"   % ipass, wmap)
			enmap.write_map(dump + "fmap_%02d.fits"   % ipass, fmap)
			enmap.write_map(dump + "norm_%02d.fits"   % ipass, norm)
		result = bunch.Bunch(snmap=fmap/norm)
		fits   = bunch.Bunch(amp=[], damp=[], pix=[], npix=[])
		# We could fit all the sources in one go, but that could lead to
		# false positives from ringing around strong sources, or lead to
		# weaker sources being masked by strong ones. So we fit in blocks
		# of source strength.
		sn_lim = np.max(fmap/norm*(apod_map>0))/snblock
		for iblock in range(nblock):
			snmap   = fmap/norm
			if dump:
				wnmap.write_map(dump + "snmap_%02d_%02d.fits" % (ipass, iblock), snmap)
			# Find all significant candidates, even those below our current block cutoff.
			# We do this because we will later compute a weighted average position, and we
			# want to use more than just a few pixels near the peak for that average.
			matches = snmap >= snmin
			labels, nlabel = ndimage.label(matches)
			if nlabel == 0: break
			all_inds = np.arange(nlabel)
			sn       = ndimage.maximum(snmap, labels, all_inds+1)
			# Then apply the sn_lim cutoff. keep is the list of matches that were sufficiently
			# strong.
			keep     = np.where(sn >= sn_lim)[0]
			if len(keep) == 0: break
			# Measure the properties of the selected sources. This will be based on the
			# pixels that were > snmin.
			pix, amp = fit_srcs(fmap, labels, keep+1)
			damp     = norm.at(pix.T, unit="pix", order=0)
			npix     = ndimage.sum(matches, labels, keep+1)
			model    = calc_model(fmap.shape, fmap.wcs, pix, amp, template)
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
				print "pass %d block %2d sn: %s" % (ipass+1, iblock+1, desc)
			# Update sn_lim. Will always decrease by at least snblock, but can go down faster
			# if there aren't any intermediate sources.
			if sn_lim <= snmin: break
			sn_lim = min(sn_lim,np.max(sn))/snblock
			# No point in continuing if we've already reached sn_lim < snmin. At this point
			# we're just digging into the noise.
		# Construct our output catalog format
		for key in fits: fits[key] = np.concatenate(fits[key])
		nsrc = len(fits.amp)
		cat = np.zeros(nsrc, cat_dtype).view(np.recarray)
		if nsrc > 0:
			rms = adiv.at(fits.pix.T, unit="pix", order=0)**-0.5
			pos = wmap.pix2sky(fits.pix.T).T
			cat.y,   cat.x  = fits.pix.T
			cat.dec, cat.ra = wmap.pix2sky(fits.pix.T)
			cat.amp  = fits.amp*rms
			cat.damp = fits.damp*rms
			cat.npix = fits.npix
			# Order by S/N
			cat = cat[np.argsort(cat.amp/cat.damp)[::-1]]
		del fits
		# Reject any sources that are in the apodization region
		dist_from_apod = ndimage.distance_transform_edt(apod_map>=1)
		untainted = dist_from_apod[utils.nint(cat.y),utils.nint(cat.x)] >= apod_margin
		cat = cat[untainted]
		# Compute model and residual in real units
		result.resid_snmap = fmap/norm
		beam_thumb  = get_thumb(enmap.ifft(beam2d+0j).real, size=kernel)
		beam_thumb /= np.max(beam_thumb)
		pix                = np.array([cat.y, cat.x]).T
		result.model       = calc_model(imap.shape, imap.wcs, pix, cat.amp, beam_thumb)
		result.resid       = imap - result.model
		result.map         = imap
		result.beam_thumb  = beam_thumb
		result.cat         = cat
		# Prepare for next iteration
		noise = result.resid
	return result

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
	pix          = np.array([result.cat.y, result.cat.x]).T
	result.model = calc_model(result.map.shape, result.map.wcs, pix, result.cat.amp, result.beam_thumb)
	result.resid = result.map - result.model
	return result

def write_catalog(ofile, cat):
	np.savetxt(ofile, np.array([
		cat.ra/utils.degree,
		cat.dec/utils.degree,
		cat.amp/cat.damp,
		cat.amp/1e3,
		cat.damp/1e3,
		cat.npix,
	]).T, fmt="%9.4f %9.4f %8.3f %9.4f %9.4f %5d")

def read_catalog(fname):
	data = np.loadtxt(fname)
	cat  = np.zeros(len(data), cat_dtype).view(np.recarray)
	cat.ra,  cat.dec  = data[:,0:2].T*utils.degree
	cat.amp, cat.damp = data[:,3:5].T*1e3
	cat.npix = data[:,5]
	return cat

def allgather_catalog(cat, comm):
	# This is hacky. It only works if all the columns of cat are floats. Which they are.
	# But it's still ugly. I wich mpi4py supported recarrays.
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
	sn     = cat.amp/cat.damp
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

def merge_duplicates(cat, rlim=1*utils.arcmin, alim=0.25):
	"""Given a point source catalog which might contain duplicates, detect these duplicates
	and merge them to produce a single catalog with no duplicates. Sources are considered
	duplicates if they are within rlim of each other. Merging uses averaging if the amplitudes
	differ by less than alim fractionally. Otherwise the strongest one is used. This is to prevent
	a strong source from being averaged with its own artifacts. rlim should be adjusted
	to fit the exerpiment beam. The default is appropriate for ACT."""
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
			amps  = cat.amp[group]
			good  = np.where(amps >= np.max(amps)*(1-alim))[0]
			gcat  = cat[group[good]]
			entry = np.zeros([], cat.dtype)
			def wmean(v, w): return np.sum(v*w)/np.sum(w)
			for key in cat.dtype.fields:
				# Weighted mean in case one is more uncertain for some reason
				entry[key] = wmean(gcat[key], gcat["damp"]**-2)
			# Set min uncertainty of inputs as the effective one. Could have
			# also used np.mean(gcat.damp**-2)**-0.5, but I trust the least uncertain
			# one more.
			entry["damp"] = np.min(gcat["damp"])
			ocat.append(entry)
			done[group] = True
	ocat = np.array(ocat).view(np.recarray)
	return ocat

def build_merge_weight(shape):
	yoff = np.arange(shape[-2])*1.0-(shape[-2]-1.0)/2
	xoff = np.arange(shape[-1])*1.0-(shape[-1]-1.0)/2
	wy   = 1-yoff/np.max(yoff)
	wx   = 1-xoff/np.max(xoff)
	weights = wy[:,None]*wx[None,:]
	return weights

def merge_maps_onto(maplist, shape, wcs, comm, root=0, crop=0):
	# First crop the maps if necessary
	if crop: maplist = [map[...,crop:-crop,crop:-crop] for map in maplist]
	if comm.rank == root:
		omap = enmap.zeros(shape, wcs)
		odiv = enmap.zeros(shape, wcs)
		for ri in range(comm.size):
			if comm.rank == ri:
				# Handle self-to-self
				for imap in maplist:
					idiv = enmap.samewcs(build_merge_weight(imap.shape),imap)
					enmap.insert(omap, imap*idiv, op=np.add)
					enmap.insert(odiv, idiv,      op=np.add)
			else:
				# Then handle those of other tasks
				nmap = comm.recv(source=ri)
				for i in range(nmap):
					pbox = np.array(comm.recv(source=ri)).reshape(2,2)
					imap = np.zeros(pbox[1]-pbox[0])
					idiv = build_merge_weight(imap.shape)
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
