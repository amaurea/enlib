""""This module pulls together all the different effects that can go into
CMB map simulations, such as curved sky, lensing, aberration, etc."""
import numpy as np
from enlib import enmap, curvedsky, lensing, aberration, utils

def rand_map(shape, wcs, ps, lmax=None, curved=True, lens=True, aberrate=True,
		beta=None, dir=None, seed=None, dtype=None, verbose=False, recenter=False,
		pad=0):
	if curved:
		return rand_map_curved(shape, wcs, ps, lmax=lmax, lens=lens, aberrate=aberrate,
				beta=beta, dir=dir, seed=seed, dtype=dtype, verbose=verbose, recenter=recenter)
	else:
		return rand_map_flat(shape, wcs, ps, lmax=lmax, lens=lens, aberrate=aberrate,
				beta=beta, dir=dir, seed=seed, dtype=dtype, verbose=verbose, recenter=recenter,
				pad=pad)

def rand_map_curved(shape, wcs, ps, lmax=None, lens=True, aberrate=True,
		beta=None, dir=None, seed=None, dtype=None, verbose=False, recenter=False):
	"""Simulate a random curved-sky map. The input spectrum should be
	[{phi,T,E,B},{phi,T,E,b},nl] of lens is True, and just [{T,E,B},{T,E,B},nl]
	otherwise."""
	if dtype is None: dtype = np.float64
	if dir   is None: dir   = aberration.dir_equ
	if beta  is None: beta  = aberration.beta
	ctype = np.result_type(dtype,0j)

	if verbose: print("Generating alms")
	alm  = curvedsky.rand_alm(ps, lmax=lmax, seed=seed, dtype=ctype)
	# Before any corrections are applied, the pixel positions map
	# directly to the raw cmb positions, and there is no induced polarization
	# rotation or amplitude modulation.
	if verbose: print("Computing observed coordinates")
	pos = enmap.posmap(shape, wcs)
	ang = enmap.zeros(shape[-2:], wcs)
	amp = enmap.zeros(shape[-2:], wcs)+1
	# Lensing remaps positions
	if lens:
		phi_alm, alm = alm[0], alm[1:]
		if verbose: print("Computing lensing gradient")
		grad = curvedsky.alm2map(phi_alm, enmap.zeros((2,)+shape[-2:], wcs, dtype=dtype), deriv=True)
		del phi_alm
		if verbose: print("Applying lensing gradient")
		pos  = enmap.samewcs(lensing.offset_by_grad(pos, grad, pol=True, geodesic=True), pos)
		ang += pos[2]
	# Aberration remaps positions and modulates amplitudes
	if aberrate:
		if verbose: print("Computing aberration")
		pos  = enmap.samewcs(aberration.remap(pos[1::-1], dir=dir, beta=beta, recenter=recenter), pos)
		ang += pos[2]
		amp *= pos[3]
		pos  = pos[1::-1]
	# Simulate the sky at the observed locations
	if verbose: print("Simulating sky signal")
	map = curvedsky.alm2map_pos(alm, pos)
	del alm, pos
	# Apply polarization rotation
	if verbose: print("Applying polarization rotation")
	map = enmap.rotate_pol(map, ang)
	# and modulation
	if verbose: print("Applying mouldation")
	map *= amp
	return map

def rand_map_flat(shape, wcs, ps, lmax=None, lens=True, aberrate=True,
		beta=None, dir=None, seed=None, dtype=None, verbose=False, recenter=False,
		pad=0):
	"""Simulate a random flat-sky map. The input spectrum should be
	[{phi,T,E,B},{phi,T,E,b},nl] of lens is True, and just [{T,E,B},{T,E,B},nl]
	otherwise."""
	if dtype is None: dtype = np.float64
	if dir   is None: dir   = aberration.dir_equ
	if beta  is None: beta  = aberration.beta
	ctype = np.result_type(dtype,0j)
	if verbose: print("Generating unlensed cmb")
	# No position calculation necessary if we're not lensing or aberrating.
	if not lens and not aberrate:
		return enmap.rand_map(shape, wcs, ps, seed=seed)
	# Otherwise we must deal with various displacements
	if aberrate: pad += np.pi*beta*1.2
	pad_pix  = int(pad / enmap.pixsize(shape, wcs)**0.5)
	if pad_pix > 0:
		if verbose: print("Padding")
		template = enmap.zeros(shape, wcs, np.int16)
		template, pslice = enmap.pad(template, pad_pix, return_slice=True)
		pshape, pwcs = template.shape, template.wcs
	else:
		pshape, pwcs = shape, wcs
	# Simulate (padded) lensing map
	if lens:
		maps = enmap.rand_map((ps.shape[0],)+pshape[-2:], pwcs, ps)
		phi, unlensed = maps[0], maps[1:]
		if verbose: print("Lensing")
		m = lensing.lens_map_flat(unlensed, phi)
	else:
		m = enmap.rand_map((ps.shape[0],)+pshape[-2:], pwcs, ps)
	# Then handle aberration if necessary
	if aberrate:
		if verbose: print("Computing aberration displacement")
		pos  = m.posmap()
		pos  = enmap.samewcs(aberration.remap(pos[1::-1], dir=dir, beta=beta, recenter=recenter), pos)
		amp  = pos[3]
		pos  = pos[1::-1]
		if verbose: print("Interpolating aberration")
		m    = enmap.samewcs(m.at(pos, mask_nan=False), m)
		if verbose: print("Applying modulation")
		m  *= amp
	if pad_pix > 0:
		if verbose: print("Unpadding")
		m = m[pslice]
	return m
