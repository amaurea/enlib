import numpy as np
from enlib import enmap, utils, powspec, sharp, curvedsky, interpol

####### Flat sky lensing #######

def lens_map(imap, grad_phi, order=3, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7):
	"""Lens map imap[{pre},ny,nx] according to grad_phi[2,ny,nx], where phi is the
	lensing potential, and grad_phi, which can be computed as enmap.grad(phi), simply
	is the coordinate displacement for each pixel. order, mode and border specify
	details of the interpolation used. See enlib.interpol.map_coordinates for details.
	If trans is true, the transpose operation is performed. This is NOT equivalent to
	delensing.

	If the same lensing field needs to be reused repeatedly, then higher efficiency
	can be gotten from calling displace_map directly with precomputed pixel positions."""
	# Converting from grad_phi to pix has roughly the same cost as calling displace_map.
	# So almost a factor 2 in speed can be won from calling displace_map directly.
	pos = imap.posmap() + grad_phi
	pix = imap.sky2pix(pos, safe=False)
	if not deriv:
		return displace_map(imap, pix, order=order, mode=mode, border=border, trans=trans)
	else:
		# displace_map deriv gives us ndim,{pre},ny,nx
		dlens_pix = displace_map(imap, pix, order=order, mode=mode, border=border, trans=trans, deriv=True)
		res = dlens_pix[0]*0
		pad = (slice(None),)+(None,)*(imap.ndim-2)+(slice(None),slice(None))
		for i in range(2):
			pos2 = pos.copy()
			pos2[i] += h
			pix2 = imap.sky2pix(pos2, safe=False)
			dpix = (pix2-pix)/h
			res += np.sum(dlens_pix * dpix[pad],0)
		return res

def delens_map(imap, grad_phi, nstep=3, order=3, mode="spline", border="cyclic"):
	"""The inverse of lens_map, such that delens_map(lens_map(imap, dpos), dpos) = imap
	for well-behaved fields. The inverse does not always exist, in which case the
	equation above will only be approximately fulfilled. The inverse is computed by
	iteration, with the number of steps in the iteration controllable through the
	nstep parameter. See enlib.interpol.map_coordinates for details on the other
	parameters."""
	grad_phi = delens_grad(grad_phi, nstep=nstep, order=order, mode=mode, border=border)
	return lens_map(imap, -grad_phi, order=order, mode=mode, border=border)

def delens_grad(grad_phi, nstep=3, order=3, mode="spline", border="cyclic"):
	"""Helper function for delens_map. Attempts to find the undisplaced gradient
	given one that has been displaced by itself."""
	alpha = grad_phi
	for i in range(nstep):
		alpha = lens_map(grad_phi, -alpha, order=order, mode=mode, border=border)
	return alpha

def displace_map(imap, pix, order=3, mode="spline", border="cyclic", trans=False, deriv=False):
	"""Displace map m[{pre},ny,nx] by pix[2,ny,nx], where pix indicates the location
	in the input map each output pixel should get its value from (float). The output
	is [{pre},ny,nx]."""
	iwork = np.empty(imap.shape[-2:]+imap.shape[:-2],imap.dtype)
	if not deriv:
		out = imap.copy()
	else:
		out = enmap.empty((2,)+imap.shape, imap.wcs, imap.dtype)
	# Why do we have to manually allocate outputs and juggle whcih is copied over here?
	# Because it is in general not possible to infer from odata what shape idata should have.
	# So the map_coordinates can't allocate the output array automatically in the transposed
	# case. But in this function we know that they will have the same shape, so we can.
	if not trans:
		iwork[:] = utils.moveaxes(imap, (-2,-1), (0,1))
		owork = interpol.map_coordinates(iwork, pix, order=order, mode=mode, border=border, trans=trans, deriv=deriv)
		out[:] = utils.moveaxes(owork, (0,1), (-2,-1))
	else:
		if not deriv:
			owork = iwork.copy()
		else:
			owork = np.empty(imap.shape[-2:]+(2,)+imap.shape[:-2],imap.dtype)
		owork[:] = utils.moveaxes(imap, (-2,-1), (0,1))
		interpol.map_coordinates(iwork, pix, owork, order=order, mode=mode, border=border, trans=trans, deriv=deriv)
		out[:] = utils.moveaxes(iwork, (0,1), (-2,-1))
	return out


# Compatibility function. Not quite equivalent lens_map above due to taking phi rather than
# its gradient as an argument.
def lens_map_flat(cmb_map, phi_map,order=4):
	gpix = enmap.grad_pix(phi_map)
        return lens_map_flat_pix(cmb_map, gpix,order=order)

def lens_map_flat_pix(cmb_map, grad_pix_map,order=4):
        """
        Helper function for lens_map_flat. Useful if grad_pix has been pre-calculated.
        """
	raw_pix  = cmb_map.pixmap() + grad_pix_map
	# And extract the interpolated values. Because of a bug in map_pixels with
	# mode="wrap", we must handle wrapping ourselves.
	npad = int(np.ceil(max(np.max(-raw_pix),np.max(raw_pix-np.array(cmb_map.shape[-2:])[:,None,None]))))
        pmap = enmap.pad(cmb_map, npad, wrap=True) if npad>0 else cmb_map
                
	return enmap.samewcs(utils.interpol(pmap, raw_pix+npad, order=order, mode="wrap"), cmb_map)



######## Curved sky lensing ########

def rand_map(shape, wcs, ps_lensinput, lmax=None, maplmax=None, dtype=np.float64, seed=None, oversample=2.0, spin=2, output="l", geodesic=True, verbose=False):
	ctype   = np.result_type(dtype,0j)
	# First draw a random lensing field, and use it to compute the undeflected positions
	if verbose: print "Computing observed coordinates"
	obs_pos = enmap.posmap(shape, wcs)
	if verbose: print "Generating alms"
	alm = curvedsky.rand_alm(ps_lensinput, lmax=lmax, seed=seed, dtype=ctype)
	phi_alm, cmb_alm = alm[0], alm[1:]
	# Truncate alm if we want a smoother map. In taylens, it was necessary to truncate
	# to a lower lmax for the map than for phi, to avoid aliasing. The appropriate lmax
	# for the cmb was the one that fits the resolution. FIXME: Can't slice alm this way.
	#if maplmax: cmb_alm = cmb_alm[:,:maplmax]
	del alm
	if "p" in output:
		if verbose: print "Computing phi map"
		phi_map = curvedsky.alm2map(phi_alm, enmap.zeros(shape[-2:], wcs, dtype=dtype))
	if verbose: print "Computing grad map"
	grad = curvedsky.alm2map(phi_alm, enmap.zeros((2,)+shape[-2:], wcs, dtype=dtype), deriv=True)
	if verbose: print "Computing alpha map"
	raw_pos = enmap.samewcs(offset_by_grad(obs_pos, grad, pol=True, geodesic=geodesic), obs_pos)
	del obs_pos, phi_alm
	if "a" not in output: del grad
	if "u" in output:
		if verbose: print "Computing unlensed map"
		cmb_raw = curvedsky.alm2map(cmb_alm, enmap.zeros(shape, wcs, dtype=dtype), spin=spin)
	if verbose: print "Computing lensed map"
	cmb_obs = curvedsky.alm2map_pos(cmb_alm, raw_pos[:2], oversample=oversample, spin=spin)
	if raw_pos.shape[0] > 2 and np.any(raw_pos[2]):
		if verbose: print "Rotating polarization"
		cmb_obs = enmap.rotate_pol(cmb_obs, raw_pos[2])
	del cmb_alm, raw_pos
	# Output in same order as specified in output argument
	res = []
	for c in output:
		if c == "l": res.append(cmb_obs)
		elif c == "u": res.append(cmb_raw)
		elif c == "p": res.append(phi_map)
		elif c == "a": res.append(grad)
	return tuple(res)

def offset_by_grad(ipos, grad, geodesic=True, pol=None):
	"""Given a set of coordinates ipos[{dec,ra},...] and a gradient
	grad[{ddec,dphi/cos(dec)},...] (as returned by curvedsky.alm2map(deriv=True)),
	returns opos = ipos + grad, while properly parallel transporting
	on the sphere. If geodesic=False is specified, then an much faster
	approximation is used, which is still very accurate unless one is
	close to the poles."""
	ncomp = 2 if pol is False or pol is None and ipos.shape[0] <= 2 else 3
	opos = np.empty((ncomp,)+ipos.shape[1:])
	iflat = ipos.reshape(ipos.shape[0],-1)
	# Oflat is a flattened view of opos, so changes to oflat
	# are visible in our return value opos
	oflat = opos.reshape(opos.shape[0],-1)
	gflat = grad.reshape(grad.shape[0],-1)
	if geodesic:
		# Loop over chunks in order to conserve memory
		step = 0x100000
		for i in range(0, iflat.shape[1], step):
			# The helper function assumes zenith coordinates
			small_grad = gflat[:,i:i+step].copy(); small_grad[0] = -small_grad[0]
			small_ipos = iflat[:,i:i+step].copy(); small_ipos[0] = np.pi/2-small_ipos[0]
			# Compute the offset position and polarization rotation, the latter in the
			# form of cos and sin of the polarization rotation angle
			small_opos, small_orot = offset_by_grad_helper(small_ipos, small_grad, ncomp>2)
			oflat[0,i:i+step] = np.pi/2 - small_opos[0]
			oflat[1,i:i+step] = small_opos[1]
			# Handle rotation if necessary
			if oflat.shape[0] > 2:
				# Recover angle from cos and sin
				oflat[2,i:i+step] = np.arctan2(small_orot[1],small_orot[0])
				if iflat.shape[0] > 2:
					oflat[2,i:i+step] += iflat[2,i:i+step]
	else:
		oflat[0] = iflat[0] + gflat[0]
		oflat[1] = iflat[1] + gflat[1]/np.cos(iflat[0])
		oflat[:2] = pole_wrap(oflat[:2])
		if oflat.shape[0] > 2: oflat[2] = 0
	return opos

def offset_by_grad_helper(ipos, grad, pol):
	"""Find the new position and induced rotation
	from offseting the input positions ipos[2,nsamp]
	by grad[2,nsamp]."""
	grad = np.array(grad)
	# Decompose grad into direction and length.
	# Fix zero-length gradients first, to avoid
	# division by zero.
	grad[:,np.all(grad==0,0)] = 1e-20
	d = np.sum(grad**2,0)**0.5
	grad  /=d
	# Perform the position offset using spherical
	# trigonometry
	cosd, sind = np.cos(d), np.sin(d)
	cost, sint = np.cos(ipos[0]), np.sin(ipos[0])
	ocost  = cosd*cost-sind*sint*grad[0]
	osint  = (1-ocost**2)**0.5
	ophi   = ipos[1] + np.arcsin(sind*grad[1]/osint)
	if not pol:
		return np.array([np.arccos(ocost), ophi]), None
	# Compute the induced polarization rotation.
	A      = grad[1]/(sind*cost/sint+grad[0]*cosd)
	nom1   = grad[0]+grad[1]*A
	denom  = 1+A**2
	cosgam = 2*nom1**2/denom-1
	singam = 2*nom1*(grad[1]-grad[0]*A)/denom
	return np.array([np.arccos(ocost), ophi]), np.array([cosgam,singam])

def pole_wrap(pos):
	"""Handle pole wraparound."""
	a = np.array(pos)
	bad = np.where(a[0] > np.pi/2)
	a[0,bad] = np.pi - a[0,bad]
	a[1,bad] = a[1,bad]+np.pi
	bad = np.where(a[0] < -np.pi/2)
	a[0,bad] = -np.pi - a[0,bad]
	a[1,bad] = a[1,bad]+np.pi
	return a

#def rand_map(shape, wcs, ps_cmb, ps_lens, lmax=None, dtype=np.float64, seed=None, oversample=2.0, spin=2, output="l", geodesic=True, verbose=False):
#	ctype   = np.result_type(dtype,0j)
#	ncomp   = ps_cmb.shape[0]
#	if ps_lens.ndim == 3: ps_lens = ps_lens[0,0]
#	# First draw a random lensing field, and use it to compute the undeflected positions
#	if verbose: print "Computing observed coordinates"
#	obs_pos = enmap.posmap(shape, wcs)
#	if verbose: print "Generating phi alms"
#	phi_alm = curvedsky.rand_alm(ps_lens, lmax=lmax, seed=seed, dtype=ctype)
#	if "p" in output:
#		if verbose: print "Computing phi map"
#		phi_map = curvedsky.alm2map_pos(phi_alm, obs_pos, oversample=oversample)
#	if verbose: print "Computing grad map"
#	grad    = curvedsky.alm2map_pos(phi_alm, obs_pos, oversample=oversample, deriv=True)
#	if verbose: print "Computing alpha map"
#	raw_pos = enmap.samewcs(offset_by_grad(obs_pos, grad, pol=ncomp>1, geodesic=geodesic), obs_pos)
#	del phi_alm
#	# Then draw a random CMB realization at the raw positions
#	if verbose: print "Generating cmb alms"
#	cmb_alm = curvedsky.rand_alm(ps_cmb, lmax=lmax, dtype=ctype) # already seeded
#	if "u" in output:
#		if verbose: print "Computing unlensed map"
#		cmb_raw = curvedsky.alm2map_pos(cmb_alm, obs_pos, oversample=oversample, spin=spin)
#	if verbose: print "Computing lensed map"
#	cmb_obs = curvedsky.alm2map_pos(cmb_alm, raw_pos[:2], oversample=oversample, spin=spin)
#	if raw_pos.shape[0] > 2 and np.any(raw_pos[2]):
#		if verbose: print "Rotating polarization"
#		cmb_obs = enmap.rotate_pol(cmb_obs, raw_pos[2])
#	del cmb_alm
#	# Output in same order as specified in output argument
#	res = []
#	for c in output:
#		if c == "l": res.append(cmb_obs)
#		elif c == "u": res.append(cmb_raw)
#		elif c == "p": res.append(phi_map)
#		elif c == "a": res.append(grad)
#	return tuple(res)
