""""This module pulls together all the different effects that can go into
CMB map simulations, such as curved sky, lensing, aberration, etc."""
import numpy as np
from enlib import enmap, curvedsky, lensing, aberration

def rand_map(shape, wcs, ps, lmax=None, lens=True, aberrate=True,
		beta=None, dir=None, seed=None, dtype=None, verbose=False, recenter=False):
	if dtype is None: dtype = np.float64
	if dir   is None: dir   = aberration.dir_equ
	if beta  is None: beta  = aberration.beta
	ctype = np.result_type(dtype,0j)

	if verbose: print "Generating alms"
	alm  = curvedsky.rand_alm(ps, lmax=lmax, seed=seed, dtype=ctype)
	# Before any corrections are applied, the pixel positions map
	# directly to the raw cmb positions, and there is no induced polarization
	# rotation or amplitude modulation.
	if verbose: print "Computing observed coordinates"
	pos = enmap.posmap(shape, wcs)
	ang = enmap.zeros(shape[-2:], wcs)
	amp = enmap.zeros(shape[-2:], wcs)+1
	# Lensing remaps positions
	if lens:
		phi_alm, alm = alm[0], alm[1:]
		if verbose: print "Computing lensing gradient"
		grad = curvedsky.alm2map(phi_alm, enmap.zeros((2,)+shape[-2:], wcs, dtype=dtype), deriv=True)
		del phi_alm
		if verbose: print "Applying lensing gradient"
		pos  = enmap.samewcs(lensing.offset_by_grad(pos, grad, pol=True, geodesic=True), pos)
		ang += pos[2]
	# Aberration remaps positions and modulates amplitudes
	if aberrate:
		if verbose: print "Computing aberration"
		pos  = enmap.samewcs(aberration.remap(pos[1::-1], dir=dir, beta=beta, recenter=recenter), pos)
		ang += pos[2]
		amp *= pos[3]
		pos  = pos[1::-1]
	# Simulate the sky at the observed locations
	if verbose: print "Simulating sky signal"
	map = curvedsky.alm2map_pos(alm, pos)
	del alm, pos
	# Apply polarization rotation
	if verbose: print "Applying polarization rotation"
	map = enmap.rotate_pol(map, ang)
	# and modulation
	if verbose: print "Applying mouldation"
	map *= amp
	return map
