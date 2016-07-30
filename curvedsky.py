"""This module provides functions for taking into account the curvature of the
full sky."""
import numpy as np
from enlib import sharp, enmap, powspec, wcs as enwcs, utils

def rand_map(shape, wcs, ps, lmax=None, dtype=np.float64, seed=None, oversample=2.0, spin=2, method="auto"):
	"""Generates a CMB realization with the given power spectrum for an enmap
	with the specified shape and WCS. This is identical to enlib.rand_map, except
	that it takes into account the curvature of the full sky. This makes it much
	slower and more memory-intensive. The map should not cross the poles."""
	# Ensure everything has the right dimensions and restrict to relevant dimensions
	ps = utils.atleast_3d(ps)
	assert ps.shape[0] == ps.shape[1], "ps must be [ncomp,ncomp,nl] or [nl]"
	assert len(shape) == 2 or len(shape) == 3, "shape must be (ncomp,ny,nx) or (ny,nx)"
	ncomp = 1 if len(shape) == 2 else shape[-3]
	ps = ps[:ncomp,:ncomp]

	ctype = np.result_type(dtype,0j)
	alm   = rand_alm(ps, lmax=lmax, seed=seed, dtype=ctype)
	map   = enmap.empty((ncomp,)+shape[-2:], wcs, dtype=dtype)
	alm2map(alm, map, spin=spin, oversample=oversample, method=method)
	if len(shape) == 2: map = map[0]
	return map

def rand_alm_healpy(ps, lmax=None, seed=None, dtype=np.complex128):
	import healpy
	if seed is not None: np.random.seed(seed)
	ps = powspec.sym_compress(ps, scheme="diag")
	return np.asarray(healpy.synalm(ps, lmax=lmax, new=True))

def rand_alm(ps, ainfo=None, lmax=None, seed=None, dtype=np.complex128, m_major=True):
	"""This is a replacement for healpy.synalm. It generates the random
	numbers in l-major order before transposing to m-major order in order
	to allow generation of low-res and high-res maps that agree on large
	scales. It uses 2/3 of the memory of healpy.synalm, and has comparable
	speed."""
	rtype = np.zeros([0],dtype=dtype).real.dtype
	ps    = np.asarray(ps)
	if ainfo is None: ainfo = sharp.alm_info(min(lmax,ps.shape[-1]-1) or ps.shape[-1]-1)
	if ps.ndim == 1:
		wps = ps[None,None]
	elif ps.ndim == 2:
		wps = powspec.sym_expand(ps, scheme="diag")
	elif ps.ndim == 3:
		wps = ps
	else:
		raise ValuerError("power spectrum must be [nl], [nspec,nl] or [ncomp,ncomp,nl]")
	ncomp = wps.shape[0]
	ps12  = enmap.multi_pow(wps, 0.5)
	# Draw random gaussian numbers in chunks to save memory
	alm   = np.empty([ncomp,ainfo.nelem],dtype=dtype)
	aflat = alm.reshape(-1).view(rtype)
	bsize = 0x10000
	if seed != None: np.random.seed(seed)
	for i in range(0, aflat.size, bsize):
		aflat[i:i+bsize] = np.random.standard_normal(min(bsize,aflat.size-i))
	# Transpose numbers to make them m-major.
	if m_major: ainfo.transpose_alm(alm,alm)
	# Scale alms by spectrum, taking into account which alms are complex
	ainfo.lmul(alm, (ps12/2**0.5).astype(rtype), alm)
	alm[:,:ainfo.lmax].imag  = 0
	alm[:,:ainfo.lmax].real *= 2**0.5
	if ps.ndim == 1: alm = alm[0]
	return alm

def alm2map(alm, map, ainfo=None, spin=2, deriv=False, direct=False, copy=False, oversample=2.0, method="auto"):
	if method == "cyl":
		alm2map_cyl(alm, map, ainfo=ainfo, spin=spin, deriv=deriv, direct=direct, copy=copy)
	elif method == "pos":
		pos = map.posmap()
		res = alm2map_pos(alm, pos, ainfo=ainfo, oversample=oversample, spin=spin, deriv=deriv)
		map[:] = res
	elif method == "auto":
		# Cylindrical method if possible, else slow pos-based method
		try:
			alm2map_cyl(alm, map, ainfo=ainfo, spin=spin, deriv=deriv, direct=direct, copy=copy)
		except AssertionError as e:
			# Wrong pixelization. Fall back on slow, general method
			pos = map.posmap()
			res = alm2map_pos(alm, pos, ainfo=ainfo, oversample=oversample, spin=spin, deriv=deriv)
			map[:] = res
	else:
		raise ValueError("Unknown alm2map method %s" % method)
	return map

def alm2map_cyl(alm, map, ainfo=None, spin=2, deriv=False, direct=False, copy=False):
	"""When called as alm2map(alm, map) projects those alms onto that map.
	alms are interpreted according to ainfo if specified.

	Possible shapes:
		alm[nelem] -> map[ny,nx]
		alm[ncomp,nelem] -> map[ncomp,ny,nx]
		alm[ntrans,ncomp,nelem] -> map[ntrans,ncomp,ny,nx]
		alm[nelem] -> map[{dy,dx},ny,nx] (deriv=True)
		alm[ntrans,nelem] -> map[ntrans,{dy,dx},ny,nx] (deriv=True)

	Spin specifies the spin of the transform. Deriv indicates whether
	we will return the derivatives rather than the map itself. If
	direct is true, the input map is assumed to already cover the whole
	sky horizontally, so that no intermediate maps need to be computed.

	If copy=True, the input map is not overwritten.
	"""
	# Work on views of alm and map with shape alm_full[ntrans,ncomp,nalm]
	# and map[ntrans,ncomp/nderiv,ny,nx] to avoid lots of if tests later.
	# We undo the reshape before returning.
	alm_full = utils.to_Nd(alm, 2 if deriv else 3)
	map_full = utils.to_Nd(map, 4)
	if ainfo is None: ainfo = sharp.alm_info(nalm=alm_full.shape[-1])
	if copy: map_full = map_full.copy()
	if direct:
		tmap, mslices, tslices = map_full, [(Ellipsis,)], [(Ellipsis,)]
	else:
		tmap, mslices, tslices = make_projectable_map_cyl(map_full)
	sht    = sharp.sht(map2minfo(tmap), ainfo)
	# We need a pixel-flattened version for the SHTs.
	tflat  = tmap.reshape(tmap.shape[:-2]+(-1,))

	# Perform the SHT
	if deriv:
		# We need alm_full[ntrans,nalm] -> tflat[ntrans,2,npix]
		# or alm_full[nalm] -> tflat[2,npix]
		tflat = sht.alm2map_der1(alm_full, tflat)
		# sharp's theta is a zenith angle, but we want a declination.
		# Actually, we may need to take into account left-handed
		# coordinates too, though I'm not sure how to detect those in
		# general.
		tflat[:,0] = -tflat[:,0]
	else:
		tflat[:,:1,:] = sht.alm2map(alm_full[:,:1,:], tflat[:,:1,:])
		if tflat.shape[1] > 1:
			tflat[:,1:,:] = sht.alm2map(alm_full[:,1:,:], tflat[:,1:,:], spin=spin)

	for mslice, tslice in zip(mslices, tslices):
		map_full[mslice] = tmap[tslice]
	return map

def alm2map_pos(alm, pos=None, ainfo=None, oversample=2.0, spin=2, deriv=False):
	"""Projects the given alms (with layout) on the specified pixel positions.
	alm[ncomp,nelem], pos[2,...] => res[ncomp,...]. It projects on a large
	cylindrical grid and then interpolates to the actual pixels. This is the
	general way of doing things, but not the fastest. Computing pos and
	interpolating takes a significant amount of time."""
	alm_full = np.atleast_2d(alm)
	if ainfo is None: ainfo = sharp.alm_info(nalm=alm_full.shape[-1])
	ashape, ncomp = alm_full.shape[:-2], alm_full.shape[-2]
	if deriv:
		# If we're computing derivatives, spin isn't allowed.
		# alm must be either [ntrans,nelem] or [nelem],
		# and the output will be [ntrans,2,ny,nx] or [2,ny,nx]
		ashape = ashape + (ncomp,)
		ncomp = 2
	tmap   = make_projectable_map(pos, ainfo.lmax, ashape+(ncomp,), oversample, alm.real.dtype)
	alm2map_cyl(alm, tmap, ainfo=ainfo, spin=spin, deriv=deriv, direct=True)
	# Project down on our final pixels. This will result in a slight smoothing
	pix = tmap.sky2pix(pos[:2])
	res = enmap.samewcs(utils.interpol(tmap, pix, mode="wrap"), pos)
	# Remove any extra dimensions we added
	if alm.ndim == alm_full.ndim-1: res = res[0]
	return res

def make_projectable_map_cyl(map):
	"""Given an enmap in a cylindrical projection, return a map with
	the same pixelization, but extended to cover a whole band in phi
	around the sky. Also returns the slice required to recover the
	input map from the output map."""
	# First check that the map has the right properties
	ny, nx = map.shape[-2:]
	vy,vx = enmap.pix2sky(map.shape, map.wcs, [np.arange(ny),np.zeros(ny)])
	hy,hx = enmap.pix2sky(map.shape, map.wcs, [np.zeros(nx),np.arange(nx)])
	dx = hx[1:]-hx[:-1]
	flip = dx[0] < 0
	if flip:
		# Sharp requires increasing phi
		map = map[...,::-1]
		dx  = np.abs(dx)
	assert np.allclose(dx,dx[0]), "Map must have constant phi spacing"
	nphi = int(np.round(2*np.pi/dx[0]))
	assert np.allclose(2*np.pi/nphi,dx[0]), "Pixels must evenly circumference"
	assert np.allclose(vx,vx[0]), "Different phi0 per row indicates non-cylindrical enmap"
	phi0 = vx[0]
	# Make a map with the same geometry covering a whole band around the sky
	# We can do this simply by extending it in the positive pixel dimension.
	oshape = map.shape[:-1]+(nphi,)
	owcs   = map.wcs
	nslice = (nx+nphi-1)/nphi
	islice, oslice = [], []
	for i in range(nslice):
		i1, i2 = i*nphi, min((i+1)*nphi,nx)
		islice.append((Ellipsis, slice(i1,i2)))
		if not flip:
			oslice.append((Ellipsis, slice(0, i2-i1)))
		else:
			# Flip back. Reverse slices are awkward
			end = nx-1-(i2-i1)
			if end < 0: end = None
			oslice.append((Ellipsis, slice(nx-1, end, -1)))
	return enmap.empty(oshape, owcs, dtype=map.dtype), islice, oslice

def make_projectable_map(pos, lmax, dims=(), oversample=2.0, dtype=float):
	"""Make a map suitable as an intermediate step in projecting alms up to
	lmax on to the given positions. Helper function for alm2map."""
	# First find the theta range of the pixels, with a 10% margin
	ra0      = np.mean(pos[1])/utils.degree
	decrange = np.array([np.min(pos[0]),np.max(pos[0])])
	decrange = (decrange-np.mean(decrange))*1.1+np.mean(decrange)
	decrange = np.array([max(-np.pi/2,decrange[0]),min(np.pi/2,decrange[1])])
	decrange /= utils.degree
	wdec = np.abs(decrange[1]-decrange[0])
	# The shortest wavelength in the alm is about 2pi/lmax. We need at least
	# two samples per mode.
	res = 180./lmax/oversample
	# Set up an intermediate coordinate system for the SHT. We will use
	# CAR coordinates conformal on the quator.
	nx,ny = int(360/res), int(wdec/res)
	wcs   = enwcs.WCS(naxis=2)
	wcs.wcs.crval = [ra0,0]
	wcs.wcs.cdelt = [360./nx,wdec/ny]
	# +1 in dec to include end points here. We do this to avoid wrapping from
	# the south pole to the north pole for full-sky maps
	wcs.wcs.crpix = [nx/2,-decrange[0]/res+1]
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	tmap = enmap.zeros(dims+(ny+1,nx),wcs)
	return tmap

def map2minfo(m):
	"""Given an enmap with constant-latitude rows and constant longitude
	intervals, return a corresponding sharp map_info."""
	theta  = np.pi/2 - m[...,:,:1].posmap(corner=False)[0,:,0]
	phi0   = m[...,:1,:1].posmap(corner=False)[1,0,0]
	nphi   = m.shape[-1]
	return sharp.map_info(theta, nphi, phi0)
