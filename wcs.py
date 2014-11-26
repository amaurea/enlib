"""This module defines shortcuts for generating WCS instances and working
with them. The bounding boxes and shapes used in this module all use
the same ordering as WCS, i.e. column major (so {ra,dec} rather than
{dec,ra}). Coordinates are assigned to pixel centers, as WCS does natively,
but bounding boxes include the whole pixels, not just their centers, which
is where the 0.5 stuff comes from."""
import numpy as np
from astropy.wcs import WCS
from enlib import slice

# Useful stuff to be able to do:
#  * Create a wcs from (point,res)
#  * Create a wcs from (box,res)
#  * Create a wcs from (box,shape)
#  * Create a wcs from (point,res,shape)
# Can support this by taking arguments:
#  pos: point[2] or box[2,2], mandatory
#  res: num or [2], optional
#  shape: [2], optional
# In cases where shape is not specified, the implied
# shape can be recovered from the wcs and a box by computing
# the pixel coordinates of the corners. So we don't need to return
# it.


#  1. Construct wcs from box, res (and return shape?)
#  2. Construct wcs from box, shape
#  3. Construct wcs from point, res (this is the most primitive version)

deg2rad = np.pi/180
rad2deg = 1/deg2rad

def car(pos, res=None, shape=None, rowmajor=False):
	"""Set up a plate carree system. See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
	w.wcs.crval = np.array([mid[0],0])
	return finalize(w, pos, res, shape)

def cea(pos, res=None, shape=None, rowmajor=False):
	"""Set up a cylindrical equal area system. See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	lam = np.cos(mid[1]*deg2rad)**2
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---CEA", "DEC--CEA"]
	w.wcs.set_pv([(2,1,lam)])
	w.wcs.crval = np.array([mid[0],0])
	return finalize(w, pos, res, shape)

def zea(pos, res=None, shape=None, rowmajor=False):
	"""Setups up an oblate Lambert's azimuthal equal area system.
	See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
	w.wcs.crval = mid
	return finalize(w, pos, res, shape)

# The airy distribution is a bit different, since is needs to
# know the size of the patch.
def air(pos, res=None, shape=None, rowmajor=False, rad=None):
	"""Setups up an Airy system. See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	if rad is None:
		if pos.ndim != 2:
			raise ValueError("Airy requires either rad or pos[2,2]")
		w = angdist(mid[0]*deg2rad,pos[0,1]*deg2rad,mid[0]*deg2rad,pos[1,1]*deg2rad)*rad2deg
		h = angdist(pos[0,0]*deg2rad,mid[1]*deg2rad,pos[1,0]*deg2rad,mid[1]*deg2rad)*rad2deg
		rad = (w+h)/4
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---AIR","DEC--AIR"]
	w.wcs.set_pv([(2,1,90-rad)])
	return finalize(w, pos, res, shape)

systems = {"car": car, "cea": cea, "air": air, "zea": zea }

def build(pos, res=None, shape=None, rowmajor=False, system="cea"):
	"""Set up the WCS system named by the "system" argument.
	pos can be either a [2] center position or a [{from,to},2]
	bounding box. At least one of res or shape must be specified.
	If res is specified, it must either be a number, in
	which the same resolution is used in each direction,
	or [2]. If shape is specified, it must be [2]. All angles
	are given in degrees."""
	return systems[system.lower()](pos, res, shape, rowmajor)

def validate(pos, res, shape, rowmajor=False):
	pos = np.asarray(pos)
	if pos.shape != (2,) and pos.shape != (2,2):
		raise ValueError("pos must be [2] or [2,2]")
	if res is None and shape is None:
		raise ValueError("Atleast one of res and shape must be specified")
	if res is not None:
		res = np.atleast_1d(res)
		if res.shape == (1,):
			res = np.array([res[0],res[0]])
		elif res.shape != (2,):
			raise ValueError("res must be num or [2]")
	if rowmajor:
		pos = pos[...,::-1]
		if shape is not None: shape = shape[::-1]
		if res is not None: res = res[::-1]
	if shape is not None:
		shape = shape[:2]
	if res is None and pos.ndim != 2:
		raise ValueError("pos must be a bounding box if res is not specified")
	mid = pos if pos.ndim == 1 else np.mean(pos,0)
	return pos, res, shape, mid

def finalize(w, pos, res, shape):
	"""Common logic for the various wcs builders. Fills in the reference
	pixel and resolution."""
	w.wcs.crpix = [0,0]
	if res is None:
		# Find the resolution that gives our box the required extent.
		w.wcs.cdelt = [1,1]
		corners = w.wcs_world2pix(pos,0)+0.5
		w.wcs.cdelt *= (corners[1]-corners[0])/shape
	else:
		w.wcs.cdelt = res
	if pos.ndim == 1:
		if shape is not None:
			# Place pixel origin at corner of shape centered on crval
			w.wcs.crpix = np.array(shape)/2+0.5
	else:
		# Make (0,0) in pixel coordinates correspond to pos[0].
		off = w.wcs_world2pix(pos[0,None],0)[0]+0.5
		w.wcs.crpix -= off
	return w

def describe(wcs):
	"""Since astropy.wcs.WCS objects do not have a useful
	str implementation, this function provides a relpacement."""
	sys  = wcs.wcs.ctype[0][-3:].lower()
	n    = wcs.naxis
	return ("%s:{cdelt:["+",".join(["%.3g"]*n)+"],crval:["+",".join(["%.3g"]*n)+"],crpix:["+",".join(["%.3g"]*n)+"]}") % ((sys,) + tuple(wcs.wcs.cdelt) + tuple(wcs.wcs.crval) + tuple(wcs.wcs.crpix))

def angdist(lon1,lat1,lon2,lat2):
	return np.arccos(np.cos(lat1)*np.cos(lat2)*(np.cos(lon1)*np.cos(lon2)+np.sin(lon1)*np.sin(lon2))+np.sin(lat1)*np.sin(lat2))

# Old stuff below

def car_old(shape, box):
	"""Set up a plate carree system with bounds
	box=[[ra0,dec0],[ra1,dec1]] and pixels in each
	direciton given by shape=[nra,ndec]. Box indicates
	the edge of pixels, not centers, so all pixels will
	be wholly inside the box."""
	box = np.asfarray(box)*rad2deg
	w = WCS(naxis=2)
	# Reference point must be on equator to
	# get an unrotated plate carree.
	w.wcs.cdelt = (box[1]-box[0])/shape[-2:]
	w.wcs.crval = np.array([0.,0.])
	# The 0.5 handles the pixel center vs. edge issue
	w.wcs.crpix = (w.wcs.crval-box[0])/w.wcs.cdelt+0.5
	w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
	return w

# cea has
# ra  = x
# dec = asin(lambda*y)
# with lambda specified by PV2_1.
# The projection will be conformal at theta_c,
# lambda = cos(theta_c)**2.
# Must find bounds in y
# y = sin(dec)/lambda
def cea_old(shape, box):
	"""Set up a cylindrical equal area system with bounds
	box=[[ra0,dec0],[ra1,dec1]] and pixels in each
	direciton given by shape=[nra,ndec]. Box indicates
	the edge of pixels, not centers, so all pixels will
	be wholly inside the box. The center of the box will be
	conformal."""
	box = np.asfarray(box)
	lam = np.cos(np.mean(box[:,1]))**2
	xybox = np.array([box[:,0],np.sin(box[:,1])/lam]).T*rad2deg
	w = WCS(naxis=2)
	w.wcs.cdelt = (xybox[1]-xybox[0])/shape[-2:]
	w.wcs.crval = np.array([0.,0.])
	# The 0.5 handles the pixel center vs. edge issue
	w.wcs.crpix = (w.wcs.crval-xybox[0])/w.wcs.cdelt+0.5
	w.wcs.set_pv([(2,1,lam)])
	w.wcs.ctype = ["RA---CEA", "DEC--CEA"]
	return w

def zea_old(shape, box):
	"""Setups up an oblate Lambert's azimuthal equal area
	system with bounds box=[[ra0,dec0],[ra1,dec1]] and pixels
	in each direciton given by shape=[nra,ndec]. Box indicates
	the edge of pixels, not centers, so all pixels will be
	wholly inside the box."""
	return autobox(shape, box, "ZEA")

def air_old(shape, box):
	"""Setups up an Airy system with bounds
	box=[[ra0,dec0],[ra1,dec1]] and pixels in each direciton
	given by shape=[nra,ndec]. Box indicates the edge of pixels,
	not centers, so all pixels will be wholly inside the box."""
	# Compute approximate ratius of box
	mra, mdec = np.mean(box,0)
	w = angdist(mra,box[0,1],mra,box[1,1])
	h = angdist(box[0,0],mdec,box[1,0],mdec)
	#d = angdist(box[0,0],box[0,1],box[1,0],box[1,1])
	rad = (w+h)/4 * 180/np.pi
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---AIR","DEC--AIR"]
	w.wcs.set_pv([(2,1,90-rad)])
	return autobox(shape, box, w)

def autobox(shape, box, name_or_wcs):
	"""Sets up a wcs with the given bounding box
	for an image of the given shape for the named
	coordinate system, provided no fancy options are
	needed. crval will be in the center of the box,
	so this function produces oblique projections."""
	box = np.asfarray(box)
	if isinstance(name_or_wcs, basestring):
		w = WCS(naxis=2)
		w.wcs.ctype = ["RA---"+name_or_wcs, "DEC--"+name_or_wcs]
	else:
		w = name_or_wcs
	# Set up temporary pixel coordinates.
	w.wcs.cdelt = np.array([1.,1.])
	w.wcs.crval = np.mean(box,0)*rad2deg
	w.wcs.crpix = np.array([0.,0.])
	corners = w.wcs_world2pix(box*rad2deg,0)+0.5
	# Shift crpix to make the corner the pixel origin
	w.wcs.crpix -= corners[0]
	# Scale cdelt so that the number of pixels inside
	# is correct
	w.wcs.crpix -= 0.5
	scale = shape[:2]/(corners[1]-corners[0])
	w.wcs.cdelt /= scale
	w.wcs.crpix *= scale
	w.wcs.crpix += 0.5
	return w

systems_old = {"car": car_old, "cea": cea_old, "air": air_old, "zea": zea_old }

def from_bounds(shape, box, system="cea"):
	"""Construct a WCS object based with bounds
	box=[[ra0,dec0],[ra1,dec1]] and pixels in each
	direciton given by shape=[nra,ndec]. Box indicates
	the edge of pixels, not centers, so all pixels will
	be wholly inside the box. The optional system
	argument determines which WCS to use. The default
	is "cea": cylindrical equal area"."""
	return systems_old[system.lower()](shape, box)
