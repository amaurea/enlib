from __future__ import division, print_function
import numpy as np, time
from . import enmap, utils, interpol, cg

def reproject(map, shape, wcs, omap=None, order=3, border="zero", rot=None,
		subsample=1, elim=1e-12, maxit=20, bsize=100, wrap="auto", margin=2,
		verbose=False, retall=False):
	"""Reproject map onto the geometry given by shape, wcs by
	approximately minimizing the difference between the spline
	representation of the input and output maps in the overlapping
	regions. Unlike enmap.project, this operation does not throw
	away statistical power when going from high to low resolution,
	but it is much slower.
	
	The spline will have the given order and boundary condition, with
	the same meaning as in enmap.interpol.map_coordinates.

	The accuracy is controlled by subsample, elim and maxit. subsample controls
	the density of points at which the integral of the squared differences of the
	input and output splines is evaluated. In my tests, subsample=1 (the default)
	results in 1e-5 residuals with a very blue spectrum. elim and maxit control
	the accuracy of the solution to the minimization.

	wrap controls whether to assume that the input map wraps.
	If it is "auto" (the default), then the input map is assumed to
	wrap at 360 degreed horizontally. Otherwise it can be controlled as
	[wy,wx] or just a number w, which gives the wrapping in each direction
	in pixels. 0 disables wrapping.

	If rot is specified, it should be a list [forwards, backwards], where
	forwards(pos) is an object that rotates [{dec,ra},...] from the
	input coordinates to output coordinates and backwards(pos) does the
	opposite rotation. This could be used to rotate from equatorial to
	galactic coordinates, for example, or to rotate to object-centered
	coordinates.

	If retall is True, then all intermediate steps in the solution process
	are returned as a list of maps.
	"""
	if omap is None: omap = enmap.zeros(map.shape[:-2]+tuple(shape), wcs, map.dtype)
	if rot  is None: rot  = (lambda x: x, lambda x:x)
	# Work with a padded coordinate system to avoid edge issues
	owork, oslice = enmap.pad(omap, margin, return_slice=True)
	shape, wcs = owork.shape, owork.wcs
	osub_shape, osub_wcs = enmap.scale_geometry(shape,     wcs,     subsample)
	isub_shape, isub_wcs = enmap.scale_geometry(map.shape, map.wcs, subsample)
	# Prepare to handle wrapping in the input map
	nphi  = utils.nint(360/np.abs(isub_wcs.wcs.cdelt[0]))
	nphi *= (nphi+isub_shape[-1]-1)//nphi
	if wrap is "auto": wrap = np.array([0,nphi])
	else: wrap = np.zeros(2,int)+wrap*subsample
	# Loop through chunks in our output map, collecting the corresponding
	# input map pixels. The variable convention here is that the first
	# letter is the coordinate system we're in, while the second one is the
	# system we belong to. E.g. iopix would be the pixel position in the
	# input map of a pixel center in the output map.
	oopos  = enmap.posmap(osub_shape, osub_wcs)
	oipos  = []
	bs    = bsize*subsample
	for y1 in range(0, oopos.shape[-2], bs):
		y2 = min(y1+bs,oopos.shape[-2])
		for x1 in range(0, oopos.shape[-1], bs):
			x2 = min(x1+bs,oopos.shape[-1])
			boopos = oopos[:,y1:y2,x1:x2]
			biopos = rot[1](boopos)
			biopix = enmap.sky2pix(isub_shape, isub_wcs, biopos)
			for i, w in enumerate(wrap):
				if w: biopix[i] = utils.rewind(biopix[i], biopix[i,0,0], period=w)
			biobox = utils.minmax(biopix,(1,2))
			# We now have the bounding box of our omap box in input pixel coordinates.
			# Crop it to the actually valid area
			for wibox, wobox in utils.sbox_wrap(biobox.T, wrap=wrap, cap=isub_shape[-2:]):
				wibox = np.array(wibox)
				iy1, ix1 = np.floor(wibox[:2,0]).astype(int)
				iy2, ix2 = np.ceil (wibox[:2,1]).astype(int)
				biipix = np.mgrid[iy1:iy2,ix1:ix2].reshape(2,-1)
				biipos = enmap.pix2sky(isub_shape, isub_wcs, biipix)
				boipos = rot[0](biipos)
				oipos.append(boipos)
	oipos = np.concatenate(oipos, -1)
	oopos = oopos.reshape(2,-1)
	opos  = np.concatenate([oopos,oipos],-1)
	del oipos, oopos
	if verbose: print("Solving for %d pixels using %d points" % (shape[-2]*shape[-1], opos.shape[-1]))
	# Phew! We finally have all our sample points!
	# Evaluate them in the input map to get our data points
	ipos = rot[1](opos)
	ipix = enmap.sky2pix(map.shape, map.wcs, ipos)
	vals = interpol.map_coordinates(map, ipix, order=order, border=border)
	del ipos, ipix
	# We're now done with the input map. The rest is just solving for the
	# output map.
	opix = enmap.sky2pix(shape, wcs, opos)
	rhs  = interpol.map_coordinates(owork*0, opix, vals, order=order, border=border, trans=True)
	def A(x):
		xmap = x.reshape(owork.shape)
		v    = interpol.map_coordinates(xmap,   opix,    order=order, border=border)
		res  = interpol.map_coordinates(xmap*0, opix, v, order=order, border=border, trans=True)
		return res.reshape(-1)
	solver = cg.CG(A, rhs.reshape(-1))
	for i in range(maxit):
		t1 = time.time()
		solver.step()
		t2 = time.time()
		if verbose:
			print("%4d %15.7e %15.7e %7.3f" % (solver.i, solver.err, np.std(solver.x), t2-t1))
		if solver.err <= elim: break
	omap[:] = solver.x.reshape(owork.shape)[oslice]
	return omap

#	# These will
#	# be oversampled representations of both maps pixels.
#	# We just use all the pixels of both for now, but to make this
#	# reasonably efficient for cases where one map is much larger
#	# than the other some smarter pruning should be used.
#	osub_shape, osub_wcs = enmap.scale_geometry(shape,     wcs,     subsample)
#	isub_shape, isub_wcs = enmap.scale_geometry(map.shape, map.wcs, subsample)
#
#	pos  = np.concatenate([
#		enmap.posmap(osub_shape, osub_wcs).reshape(2,-1),
#		enmap.posmap(isub_shape, isub_wcs).reshape(2,-1)], -1)
#	opix = enmap.sky2pix(shape, wcs, pos)
#	# Get rid of points that don't hit our output or input maps
#	mask = np.all(opix >= 0-margin,0) & np.all(opix < np.array(shape[-2:])[:,None]+margin,0)
#	pos, opix = pos[:,mask], opix[:,mask]
#	ipix = enmap.sky2pix(map.shape, map.wcs, pos)
#	mask = np.all(ipix >= 0-margin,0) & np.all(ipix < np.array(map.shape[-2:])[:,None]+margin,0)
#	pos, opix, ipix = pos[:,mask], opix[:,mask], ipix[:,mask]
#	# Evaluate the input map at our points. This is the only point where the
#	# input map values are used.
#	vals = interpol.map_coordinates(map, ipix, order=order, border=border)
#	# Will now minimize chisq = (vals - Pa)'(vals - Pa), where a are the
#	# omap values and P is the interpolation operation. This is minimized by
#	# P'Pa = P'vals, which we solve using conjugate gradients.
#	rhs  = interpol.map_coordinates(omap*0, opix, vals, order=order, border=border, trans=True)
#	def A(x):
#		xmap = x.reshape(omap.shape)
#		v    = interpol.map_coordinates(xmap,   opix,    order=order, border=border)
#		res  = interpol.map_coordinates(xmap*0, opix, v, order=order, border=border, trans=True)
#		return res.reshape(-1)
#	solver = cg.CG(A, rhs.reshape(-1))
#	for i in range(maxit):
#		t1 = time.time()
#		solver.step()
#		t2 = time.time()
#		if verbose:
#			print("%4d %15.7e %15.7e %7.3f" % (solver.i, solver.err, np.std(solver.x), t2-t1))
#		if solver.err <= elim: break
#	omap[:] = solver.x.reshape(omap.shape)
#	return omap
