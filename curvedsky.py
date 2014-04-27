"""This module provides functions for taking into account the curvature of the
full sky."""
import numpy as np
from enlib import sharp, enmap, powspec, wcs, utils

def rand_map(shape, wcs, ps, lmax=None, dtype=np.float64, seed=None, oversample=2.0, spin=2):
	"""Generates a CMB realization with the given power spectrum for an enmap
	with the specified shape and WCS. This is identical to enlib.rand_map, except
	that it takes into account the curvature of the full sky. This makes it much
	slower and more memory-intensive. The map should not cross the poles."""
	ctype = np.result_type(dtype,0j)
	alm   = rand_alm(ps, lmax=lmax, seed=seed, dtype=ctype)
	# Now find the pixels to project on
	pos   = enmap.posmap(shape,wcs)
	return enmap.samewcs(alm2map(alm, pos, oversample=oversample, spin=spin), wcs)

def rand_alm(ps, ainfo=None, lmax=None, seed=None, dtype=np.complex128):
	"""This is a replacement for healpy.synalm. It generates the random
	numbers in l-major order before transposing to m-major order in order
	to allow generation of low-res and high-res maps that agree on large
	scales. It uses 2/3 of the memory of healpy.synalm, and has comparable
	speed."""
	rtype = np.zeros([0],dtype=dtype).real.dtype
	ps    = np.asarray(ps)
	if ainfo is None: ainfo = sharp.alm_info(lmax or ps.shape[-1]-1)
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
	ainfo.transpose_alm(alm,alm)
	# Scale alms by spectrum, taking into account which alms are complex
	ainfo.lmul(alm, (ps12/2**0.5).astype(rtype), alm)
	alm[:,:ainfo.lmax].imag  = 0
	alm[:,:ainfo.lmax].real *= 2**0.5
	if ps.ndim == 1: alm = alm[0]
	return alm

def alm2map(alm, pos, ainfo=None, oversample=2.0, spin=2, deriv=False):
	"""Projects the given alms (with layout) on the specified pixel positions.
	alm[ncomp,nelem], pos[2,...] => res[ncomp,...]."""
	alm_full = np.atleast_2d(alm)
	if ainfo is None: ainfo = sharp.alm_info(nalm=alm_full.shape[-1])
	ashape, ncomp = alm_full.shape[:-2], alm_full.shape[-2]
	if deriv: ncomp = 2
	tmap   = make_projectable_map(pos, ainfo.lmax, ashape+(ncomp,), oversample, alm.real.dtype)
	sht    = sharp.sht(map2minfo(tmap), ainfo)
	# We need a pixel-flattened version for the SHTs
	tflat  = tmap.reshape(tmap.shape[:-2]+(-1,))

	# Perform the SHT
	if deriv:
		tflat = sht.alm2map_der1(alm_full, tflat)
		# sharp's theta is a zenith angle, but we want a declination
		tflat[...,0] = -tflat[...,0]
	else:
		tflat[...,:1,:] = sht.alm2map(alm_full[...,:1,:], tflat[...,:1,:])
		if ncomp > 1:
			tflat[...,1:,:] = sht.alm2map(alm_full[...,1:,:], tflat[...,1:,:], spin=spin)
	# Make output have the same shape as input, if possible
	if alm.ndim == 1 and tmap.shape[-3] == 1:
		tmap = tmap[...,0,:,:]

	# Project down on our final pixels. This will result in a slight smoothing
	pix = tmap.sky2pix(pos[:2])
	return enmap.samewcs(utils.interpol(tmap, pix), pos)

def make_projectable_map(pos, lmax, dims=(), oversample=2.0, dtype=float):
	"""Make a map suitable as an intermediate step in projecting alms up to
	lmax on to the given positions. Helper function for alm2map."""
	# First find the theta range of the pixels, with a 10% margin
	decrange = np.array([np.max(pos[0]),np.min(pos[0])])
	decrange = (decrange-np.mean(decrange))*1.1+np.mean(decrange)
	decrange = np.array([max(0,decrange[0]),min(np.pi,decrange[1])])
	# The shortest wavelength in the alm is about 2pi/lmax. We need at least
	# two samples per mode.
	step = np.pi/lmax/oversample
	# Set up an intermediate coordinate system for the SHT
	tbox   = np.array([[decrange[0],-np.pi],[decrange[1],np.pi]])
	tshape = tuple(np.ceil(np.abs((tbox[1]-tbox[0])/step)).astype(int))
	twcs   = wcs.cea(tshape[::-1], tbox[:,::-1])
	tmap   = enmap.zeros(dims+tshape, twcs, dtype=dtype)
	return tmap

def map2minfo(m):
	"""Given an enmap with constant-latitude rows and constant longitude
	intervals, return a corresponding sharp map_info."""
	theta  = np.pi/2 - m[...,:,:1].posmap(center=True)[0,:,0]
	phi0   = m[...,:1,:1].posmap(center=True)[1,0,0]
	nphi   = m.shape[-1]
	return sharp.map_info(theta, nphi, phi0)
