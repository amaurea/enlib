"""This module provides conversions between astronomical coordinate systems.
When c is more developed, it might completely replace this
module. For now, it is used as a part of the implementation."""
import numpy as np, pyfsla
import astropy.coordinates as c, astropy.units as u, ephem
from enlib import iers, utils
from enlib.utils import ang2rect, rect2ang
try:
	from pyslalib import slalib
except ImportError:
	pass

class default_site:
	lat  = -22.9585
	lon  = -67.7876
	alt  = 5188.
	T    = 273.15
	P    = 550.
	hum  = 0.2
	freq = 150.
	lapse= 0.0065

def transform(from_sys, to_sys, coords, time=55500, site=default_site, pol=None, mag=None):
	"""Transforms coords[2,...] from system from_sys to system to_sys, where
	systems can be "hor", "cel" or "gal". For transformations involving
	"hor", the optional arguments time (in modified julian days) and site (which must
	contain .lat (rad), .lon (rad), .P (pressure, mBar), .T (temperature, K),
	.hum (humidity, 0.2 by default), .alt (altitude, m)). Returns an array
	with the same shape as the input. The coordinates are in ra,dec-ordering."""
	from_info, to_info = getsys_full(from_sys,time,site), getsys_full(to_sys,time,site)
	ihand = get_handedness(from_info[0])
	ohand = get_handedness(to_info[0])
	# Apply the specified transformation, optionally computing the induced
	# polarization rotation and apparent magnification
	def transfunc(coords):
		return transform_raw(from_info, to_info, coords, time=time, site=site)
	fields = []
	if pol: fields.append("ang")
	if mag: fields.append("mag")
	if pol is None and mag is None:
		if len(coords) > 2: fields.append("ang")
		if len(coords) > 3: fields.append("mag")
	meta = transform_meta(transfunc, coords[:2], fields=fields)

	# Fix the polarization convention. We use healpix
	if "ang" in fields:
		if ihand != ohand: meta.ang -= np.pi
		if ohand != 'L':   meta.ang = -meta.ang
	# Create the output array. This is a bit cumbersome because
	# each of the output columns can be either ang or mag, which
	# might or might not have previous values that need to be
	# updated. It is this way to keep backward compatibility.
	res = np.zeros((2+len(fields),) + meta.ocoord.shape[1:])
	res[:2] = meta.ocoord
	off = 2
	for i, f in enumerate(fields):
		if f == "ang":
			if len(coords) > 2: res[off+i] = coords[2] + meta.ang
			else: res[off+i] = meta.ang
		elif f == "mag":
			if len(coords) > 3: res[off+i] = coords[3] * meta.mag
			else: res[off+i] = meta.mag
	return res

def transform_meta(transfun, coords, fields=["ang","mag"], offset=5e-7):
	"""Computes metadata for the coordinate transformation functor
	transfun applied to the coordinate array coords[2,...],
	such as the induced rotation, magnification.

	Currently assumes that input and output coordinates are in
	non-zenith polar coordinates. Might generalize this later.
	"""
	if "mag_brute" in fields: ntrans = 3
	elif "ang" in fields: ntrans = 2
	else: ntrans = 1
	coords  = np.asarray(coords)
	offsets = np.array([[0,0],[1,0],[0,1]])*offset
	# Transform all the coordinates. We assume we aren't super-close to the poles
	# either before or after the transformation.
	ocoords = np.zeros((ntrans,2)+coords.shape[1:])
	ocoords = None
	for i in range(ntrans):
		# Transpose to get broadcasting right
		a = transfun((coords.T + offsets[i].T).T)
		if ocoords is None:
			ocoords = np.zeros((ntrans,)+a.shape, a.dtype)
		ocoords[i] = a

	class Result: pass
	res = Result()
	res.icoord = coords
	res.ocoord = ocoords[0]

	# Compute the individual properties we're interested in
	diff = utils.rewind(ocoords[1:]-ocoords[0,None])
	if "ang" in fields:
		# We only need the theta offset of this one. We started with
		# an offset in the [1,0] direction, and want to know how
		# far we have rotated away from this direction. This
		# Uses the IAU tangent plane angle convention:
		# http://healpix.jpl.nasa.gov/html/intronode12.htm
		# and assumes that both input and putput coordinates have the
		# same handedness. This is not always the case, for example
		# with horizontal to celestial coordinate transformations.
		# In these cases, the caller must correct there resulting angle
		# manually.
		phiscale = np.cos(ocoords[0,1])
		res.ang = np.arctan2(diff[0,1],diff[0,0]*phiscale)
	if "mag" in fields:
		res.mag = np.cos(res.icoord[1])/np.cos(res.ocoord[1])
	if "mag_brute" in fields:
		# Compute the ratio of the areas of the triangles
		# made up by the three point-sets in the input and
		# output coordinates. This ratio is always 1 when
		# using physical areas, so we instead compute the
		# apparent areas here.
		def tri_area(diff):
			return 0.5*np.abs(diff[0,0]*diff[1,1]-diff[0,1]*diff[1,0])
		res.mag = (tri_area(diff).T/tri_area(offsets[1:]-offsets[0]).T).T
	return res

def transform_raw(from_sys, to_sys, coords, time=None, site=None):
	"""Transforms coords[2,...] from system from_sys to system to_sys, where
	systems can be "hor", "cel" or "gal". For transformations involving
	"hor", the optional arguments time (in modified julian days) and site (which must
	contain .lat (rad), .lon (rad), .P (pressure, mBar), .T (temperature, K),
	.hum (humidity, 0.2 by default), .alt (altitude, m)). Returns an array
	with the same shape as the input. The coordinates are in ra,dec-ordering.

	coords and time will be broadcast such that the result has the same shape
	as coords*time[None]."""
	# Prepare input and output arrays
	if time is None:
		coords = np.array(coords)[:2]
	else:
		time   = np.asarray(time)
		coords = np.asarray(coords)
		# Broadasting. A bit complicated because we want to handle
		# both time needing to broadcast and coords needing to
		time   = time + np.zeros(coords[0].shape,time.dtype)
		coords = (coords.T + np.zeros(time.shape,coords.dtype)[None].T).T
	# flatten, so the rest of the code can assume that coordinates are [2,N]
	# and time is [N]
	oshape = coords.shape
	coords= np.ascontiguousarray(coords.reshape(2,-1))
	if time is not None: time = time.reshape(-1)
	# Perform the actual coordinate transformation. There are three classes of
	# transformations here:
	# 1. To/from object-centered coordinates
	# 2. cel-hor transformation, using slalib
	# 3. cel-gal transformation, using astropy
	(from_sys,from_ref), (to_sys,to_ref) = getsys_full(from_sys,time,site), getsys_full(to_sys,time,site)
	if from_ref is not None: coords[:] = decenter(coords, from_ref)
	if from_sys != to_sys:
		if from_sys == c.AltAz:
			coords[:] = hor2cel(coords, time, site, copy=False)
		coords[:] = transform_astropy(nohor(from_sys), nohor(to_sys), coords)
		if to_sys == c.AltAz:
			coords[:] = cel2hor(coords, time, site, copy=False)
	if to_ref is not None: coords[:] = recenter(coords, to_ref)
	return coords.reshape(oshape)

def transform_astropy(from_sys, to_sys, coords):
	"""As transform, but only handles the systems supported by astropy."""
	from_sys, to_sys = getsys(from_sys), getsys(to_sys)
	if from_sys == to_sys: return coords
	unit   = u.radian
	coords = c.SkyCoord(coords[0], coords[1], frame=from_sys, unit=unit)
	coords = coords.transform_to(to_sys)
	names  = coord_names[to_sys]
	return np.asarray([
		getattr(getattr(coords, names[0]),unit.name),
		getattr(getattr(coords, names[1]),unit.name)])

def hor2cel(coord, time, site, copy=True):
	coord  = np.array(coord, copy=copy)
	trepr  = time[len(time)/2]
	info   = iers.lookup(trepr)
	ao = slalib.sla_aoppa(trepr, info.dUT, site.lon*utils.degree, site.lat*utils.degree, site.alt,
		info.pmx*utils.arcsec, info.pmy*utils.arcsec, site.T, site.P, site.hum,
		299792.458/site.freq, site.lapse)
	am = slalib.sla_mappa(2000.0, trepr)
	# This involves a transpose operation, which is not optimal
	pyfsla.aomulti(time, coord.T, ao, am)
	return coord

def cel2hor(coord, time, site, copy=True):
	# This is very slow for objects near the horizon!
	coord  = np.array(coord, copy=copy)
	trepr  = time[len(time)/2]
	info   = iers.lookup(trepr)
	ao = slalib.sla_aoppa(trepr, info.dUT, site.lon*utils.degree, site.lat*utils.degree, site.alt,
		info.pmx*utils.arcsec, info.pmy*utils.arcsec, site.T, site.P, site.hum,
		299792.458/site.freq, site.lapse)
	am = slalib.sla_mappa(2000.0, trepr)
	# This involves a transpose operation, which is not optimal
	pyfsla.oamulti(time, coord.T, ao, am)
	return coord

def euler_mat(euler_angles, kind="zyz"):
	"""Defines the rotation matrix M for a ABC euler rotation,
	such that M = A(alpha)B(beta)C(gamma), where euler_angles =
	[alpha,beta,gamma]. The default kind is ABC=ZYZ."""
	alpha, beta, gamma = euler_angles
	R1 = utils.rotmatrix(gamma, kind[2])
	R2 = utils.rotmatrix(beta,  kind[1])
	R3 = utils.rotmatrix(alpha, kind[0])
	return np.einsum("...ij,...jk->...ik",np.einsum("...ij,...jk->...ik",R3,R2),R1)

def euler_rot(euler_angles, coords, kind="zyz"):
	coords = np.asarray(coords)
	co     = coords.reshape(2,-1)
	M      = euler_mat(euler_angles, kind)
	rect   = ang2rect(co, False)
	rect   = np.einsum("...ij,j...->i...",M,rect)
	co     = rect2ang(rect, False)
	return co.reshape(coords.shape)

def recenter(angs, center):
	"""Recenter coordinates "angs" (as ra,dec) on the location given by "center",
	such that center moves to the north pole."""
	# Performs the rotation E(0,-theta,-phi). Originally did
	# E(phi,-theta,-phi), but that is wrong (at least for our
	# purposes), as it does not preserve the relative orientation
	# between the boresight and the sun. For example, if the boresight
	# is at the same elevation as the sun but 10 degrees higher in az,
	# then it shouldn't matter what az actually is, but with the previous
	# method it would.
	#
	# Now supports specifying where to recenter by specifying center as
	# lon_from,lat_from,lon_to,lat_to
	if len(center) == 4: ra0, dec0, ra1, dec1 = center
	elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], 0, np.pi/2
	return euler_rot([ra1,dec0-dec1,-ra0], angs, kind="zyz")
def decenter(angs, center):
	"""Inverse operation of recenter."""
	if len(center) == 4: ra0, dec0, ra1, dec1 = center
	elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], 0, np.pi/2
	return euler_rot([ra0,dec1-dec0,-ra1],  angs, kind="zyz")

def nohor(sys): return sys if sys != "altaz" else "icrs"
def getsys(sys): return str2sys[sys.lower()] if isinstance(sys,basestring) else sys
def get_handedness(sys):
	"""Return the handedness of the coordinate system sys, as seen from inside
	the celestial sphere, in the standard IAU convention."""
	if sys in ["altaz"]: return 'R'
	else: return 'L'

def getsys_full(sys, time=None, site=None):
	"""Handles our expanded coordinate system syntax: base[:ref[:refsys]].
	This allows a system to be recentered on a given position or object.
	The argument can either be a string of the above format (with [] indicating
	optional parts), or a list of [base, ref, refsys]. Returns a parsed
	and expanded version, where the systems have been replaced by full
	system objects (or None), and the reference point has been expanded
	into coordinates (or None), and rotated into the base system."""
	if isinstance(sys, basestring): sys = sys.split(":")
	else:
		try: sys = list(sys)
		except TypeError: sys = [sys]
	if len(sys) < 3: sys += [None]*(3-len(sys))
	base, ref, refsys = sys
	base   = getsys(base)
	refsys = getsys(refsys) if refsys is not None else base
	if ref is None: return [base, ref]
	if isinstance(ref, basestring):
		# The general formt here is from[/to], where from and to
		# each are either an object name or a position in the format
		# lat_lon. comma would have been preferable, but we reserve that
		# for from_sys,to_sys uses for backwards compatibility with
		# existing programs.
		ref_expanded = []
		for r in ref.split("/"):
			# In our first format, ref is a set of coordinates in degrees
			try:
				r = np.asfarray(r.split("_"))*utils.degree
				assert(r.ndim == 1 and len(r) == 2)
				r = transform_raw(refsys, base, r[:,None], time=time, site=site)
			except ValueError:
				# Otherwise, treat as an ephemeris object
				r = ephem_pos(r, time)
				r = transform_raw("equ", base, r, time=time, site=site)
			ref_expanded += list(r)
		ref = np.array(ref_expanded)
	return [base, ref]

def ephem_pos(name, mjd):
	"""Given the name of an ephemeris object from pyephem and a
	time in modified julian date, return its position in ra, dec
	in radians in equatorial coordinates."""
	mjd = np.asarray(mjd)
	djd = mjd + 2400000.5 - 2415020
	obj = getattr(ephem, name)()
	if mjd.ndim == 0:
		obj.compute(djd)
		return np.array([float(obj.a_ra), float(obj.a_dec)])
	else:
		res = np.empty((2,djd.size))
		for i, t in enumerate(djd.reshape(-1)):
			obj.compute(t)
			res[0,i] = float(obj.a_ra)
			res[1,i] = float(obj.a_dec)
		return res.reshape((2,)+djd.shape)

def interpol_pos(from_sys, to_sys, name_or_pos, mjd, site=None, dt=10):
	"""Given the name of an ephemeris object or a [ra,dec]-type position
	in radians in from_sys, compute its position in the specified coordinate system for
	each mjd. The mjds are assumed to be sampled densely enough that
	interpolation will work. For ephemeris objects, positions are
	computed in steps of 10 seconds by default (controlled by the dt argument)."""
	box  = utils.widen_box([np.min(mjd),np.max(mjd)], 1e-2)
	sub_nsamp = max(3,int((box[1]-box[0])*24.*3600/dt))
	sub_mjd = np.linspace(box[0], box[1], sub_nsamp, endpoint=True)
	if isinstance(name_or_pos, basestring):
		sub_from = ephem_pos(name_or_pos, sub_mjd)
	else:
		pos = np.asarray(name_or_pos)
		assert pos.ndim == 1
		sub_from = np.zeros([2,sub_nsamp])
		sub_from[:] = np.asarray(name_or_pos)[:,None]
	sub_pos = transform_raw(from_sys, to_sys, sub_from, time=sub_mjd, site=site)
	sub_pos[1] = utils.rewind(sub_pos[1], ref="auto")
	inds = (mjd-box[0])*(sub_nsamp-1)/(box[1]-box[0])
	full_pos= utils.interpol(sub_pos, inds[None], order=3)
	return full_pos

def make_mapping(dict): return {value:key for key in dict for value in dict[key]}
str2sys = make_mapping({
	"galactic": ["gal", "galactic"],
	"icrs":     ["equ", "equatorial", "cel", "celestial", "icrs"],
	"altaz":    ["altaz", "azel", "hor", "horizontal"],
	"barycentrictrueecliptic": ["ecl","ecliptic","barycentrictrueecliptic"]})
coord_names = {
	"galactic": ["l","b"],
	"icrs": ["ra","dec"],
	"altaz":["az","alt"],
	"barycentrictrueecliptic":["lon","lat"]
	}
