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

def transform(from_sys, to_sys, coords, unit="rad", time=None, site=None, pol=None):
	"""Transforms coords[2,N] from system from_sys to system to_sys, where
	systems can be "hor", "cel" or "gal". For transformations involving
	"hor", the optional arguments time (in modified julian days) and site (which must
	contain .lat (rad), .lon (rad), .P (pressure, mBar), .T (temperature, K),
	.hum (humidity, 0.2 by default), .alt (altitude, m)). Returns an array
	with the same shape as the input. The coordinates are in ra,dec-ordering."""
	# Make ourselves case insensitive, and look up the corresponding objects
	unit = getunit(unit)
	coords = np.ascontiguousarray(np.array(coords))
	if time is not None: time = np.asarray(time)
	(from_sys,from_ref), (to_sys,to_ref) = getsys_full(from_sys,unit,time,site), getsys_full(to_sys,unit,time,site)
	# Handle polarization by calling ourselves twice wtih slightly differing positions.
	if pol is None: pol = len(coords) == 3
	if pol:
		coord2 = np.array(coords,dtype=float)
		coord2[0] += 0.1/60/60 / unit.in_units(u.deg)
		ocoord1 = transform((from_sys,from_ref), (to_sys,to_ref), coords[:2], unit, time, site, pol=False)
		ocoord2 = transform((from_sys,from_ref), (to_sys,to_ref), coord2[:2], unit, time, site, pol=False)
		diff = utils.rewind(ocoord2-ocoord1, 0, 360/unit.in_units(u.deg))
		ocoord  = np.empty((3,)+coord2.shape[1:])
		ocoord[:2] = ocoord1
		# The polarization rotation is defined in the tangent plane of the point,
		# so we must scale the phi coordinate to account for the sphere's curvature.
		# We assume theta to be measured from the equator for this, i.e. not a
		# zenith angle.
		phiscale   = np.cos(ocoord1[1]*unit.in_units(u.rad))
		ocoord[2]  = np.arctan2(diff[1],diff[0]*phiscale) / unit.in_units(u.rad)
		if len(coords) >= 3: ocoord[2] += coords[2]
		# We use the HEALPix left-handed polarization convention, so take that
		# into account
		ihand = get_handedness(from_sys)
		ohand = get_handedness(to_sys)
		if ihand != ohand:
			ocoord[2] = ocoord[2]-np.pi / unit.in_units(u.rad)
		if ohand != 'L':
			ocoord[2] = -ocoord[2]
		return ocoord
	if from_ref is not None: coords = decenter(coords, from_ref, unit)
	if from_sys != to_sys:
		if from_sys == c.AltAz:
			if unit != u.rad: coords = coords * unit.in_units(u.rad)
			coords = hor2cel(coords, time, site, copy=False)
			if unit != u.rad: coords = coords / unit.in_units(u.rad)
		coords = transform_astropy(nohor(from_sys), nohor(to_sys), coords, unit)
		if to_sys == c.AltAz:
			if unit != u.rad: coords = coords * unit.in_units(u.rad)
			coords = cel2hor(coords, time, site, copy=False)
			if unit != u.rad: coords = coords / unit.in_units(u.rad)
	if to_ref is not None: coords = recenter(coords, to_ref, unit)
	return coords

def transform_astropy(from_sys, to_sys, coords, unit):
	"""As transform, but only handles the systems supported by astropy."""
	unit, from_sys, to_sys = getunit(unit), getsys(from_sys), getsys(to_sys)
	if from_sys == to_sys: return coords
	coords = from_sys(coords[0], coords[1], unit=(unit,unit))
	coords = coords.transform_to(to_sys)
	names  = coord_names[to_sys]
	return np.asarray([
		getattr(getattr(coords, names[0]),unit.name),
		getattr(getattr(coords, names[1]),unit.name)])

def hor2cel(coord, time, site, copy=True):
	coord  = np.array(coord, copy=copy)
	trepr  = time[len(time)/2]
	info   = iers.lookup(trepr)
	as2rad = np.pi/180/60/60
	ao = slalib.sla_aoppa(trepr, info.dUT, site.lon*np.pi/180, site.lat*np.pi/180, site.alt,
		info.pmx*as2rad, info.pmy*as2rad, site.T, site.P, site.hum,
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
	as2rad = np.pi/180/60/60
	ao = slalib.sla_aoppa(trepr, info.dUT, site.lon*np.pi/180, site.lat*np.pi/180, site.alt,
		info.pmx*as2rad, info.pmy*as2rad, site.T, site.P, site.hum,
		299792.458/site.freq, site.lapse)
	am = slalib.sla_mappa(2000.0, trepr)
	# This involves a transpose operation, which is not optimal
	pyfsla.oamulti(time, coord.T, ao, am)
	return coord

def rotmatrix(ang, axis, unit):
	unit = getunit(unit)
	ang  = np.asarray(ang)
	axis = axis.lower()
	if unit != u.rad: ang = ang * unit.in_units(u.rad)
	c, s = np.cos(ang), np.sin(ang)
	R = np.zeros(ang.shape + (3,3))
	if   axis == "x": R[...,0,0]=1;R[...,1,1]= c;R[...,1,2]=-s;R[...,2,1]= s;R[...,2,2]=c
	elif axis == "y": R[...,0,0]=c;R[...,0,2]= s;R[...,1,1]= 1;R[...,2,0]=-s;R[...,2,2]=c
	elif axis == "z": R[...,0,0]=c;R[...,0,1]=-s;R[...,1,0]= s;R[...,1,1]= c;R[...,2,2]=1
	else: raise ValueError("Axis %s not recognized" % axis)
	return R

def euler_mat(euler_angles, kind="zyz", unit="rad"):
	"""Defines the rotation matrix M for a ABC euler rotation,
	such that M = A(alpha)B(beta)C(gamma), where euler_angles =
	[alpha,beta,gamma]. The default kind is ABC=ZYZ."""
	unit = getunit(unit)
	alpha, beta, gamma = euler_angles
	R1 = rotmatrix(gamma, kind[2], unit)
	R2 = rotmatrix(beta,  kind[1], unit)
	R3 = rotmatrix(alpha, kind[0], unit)
	return np.einsum("...ij,...jk->...ik",np.einsum("...ij,...jk->...ik",R3,R2),R1)

def euler_rot(euler_angles, coords, kind="zyz", unit="rad"):
	unit   = getunit(unit)
	coords = np.asarray(coords)
	co     = coords.reshape(2,-1)
	if unit != u.rad: co = co * unit.in_units(u.rad)
	M      = euler_mat(euler_angles, kind, unit)
	rect   = ang2rect(co, False)
	rect   = np.einsum("...ij,j...->i...",M,rect)
	co     = rect2ang(rect, False)
	if unit != u.rad: co = co / unit.in_units(u.rad)
	return co.reshape(coords.shape)

def recenter(angs, center, unit="rad"):
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
	unit = getunit(unit)
	if len(center) == 4: ra0, dec0, ra1, dec1 = center
	elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], 0, 90/unit.in_units(u.deg)
	return euler_rot([ra1,dec0-dec1,-ra0], angs, kind="zyz", unit=unit)
def decenter(angs, center, unit="rad"):
	"""Inverse operation of recenter."""
	unit = getunit(unit)
	if len(center) == 4: ra0, dec0, ra1, dec1 = center
	elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], 0, 90/unit.in_units(u.deg)
	return euler_rot([ra0,dec1-dec0,-ra1],  angs, kind="zyz", unit=unit)

def nohor(sys): return sys if sys != c.AltAz else c.ICRS
def getsys(sys): return str2sys[sys.lower()] if isinstance(sys,basestring) else sys
def getunit(u): return str2unit[u.lower()] if isinstance(u,basestring) else u
def get_handedness(sys):
	"""Return the handedness of the coordinate system sys, as seen from inside
	the celestial sphere, in the standard IAU convention."""
	if sys in [c.AltAz]: return 'R'
	else: return 'L'

def getsys_full(sys, unit="deg", time=None, site=None):
	"""Handles our expanded coordinate system syntax: base[:ref[:refsys]].
	This allows a system to be recentered on a given position or object.
	The argument can either be a string of the above format (with [] indicating
	optional parts), or a list of [base, ref, refsys]. Returns a parsed
	and expanded version, where the systems have been replaced by full
	system objects (or None), and the reference point has been expanded
	into coordinates (or None), and rotated into the base system."""
	unit = getunit(unit)
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
				r = np.asfarray(r.split("_"))/unit.in_units(u.deg)
				assert(r.ndim == 1 and len(r) == 2)
				r = transform(refsys, base, r, unit=unit, time=time, site=site)
			except ValueError:
				# Otherwise, treat as an ephemeris object
				r = ephem_pos(r, time)/unit.in_units(u.rad)
				r = transform("equ", base, r, unit=unit, time=time, site=site)
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
		return np.array([float(obj.ra), float(obj.dec)])
	else:
		res = np.empty((2,djd.size))
		for i, t in enumerate(djd.reshape(-1)):
			obj.compute(t)
			res[0,i] = float(obj.ra)
			res[1,i] = float(obj.dec)
		return res.reshape((2,)+djd.shape)

def interpol_pos(from_sys, to_sys, name_or_pos, mjd, site=None, dt=10):
	"""Given the name of an ephemeris object or a [ra,dec]-type position
	in radians in from_sys, compute its position in the specified coordinate system for
	each mjd. The mjds are assumed to cover a short
	enough range that positions can be effectively
	interpolated."""
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
	sub_pos = transform(from_sys, to_sys, sub_from, time=sub_mjd, site=site)
	sub_pos[1] = utils.rewind(sub_pos[1], ref="auto")
	inds = (mjd-box[0])*(sub_nsamp-1)/(box[1]-box[0])
	full_pos= utils.interpol(sub_pos, inds[None], order=3)
	return full_pos

def make_mapping(dict): return {value:key for key in dict for value in dict[key]}
str2unit = make_mapping({
	u.radian: ["rad", "radian", "radians"],
	u.degree: ["deg", "degree", "degrees"],
	u.arcmin: ["min", "arcmin", "arcmins"],
	u.arcsec: ["sec", "arcsec", "arcsecs"]})
str2sys = make_mapping({
	c.Galactic: ["gal", "galactic"],
	c.ICRS:     ["equ", "equatorial", "cel", "celestial", "icrs"],
	c.AltAz:    ["altaz", "azel", "hor", "horizontal"]})
coord_names = {
	c.Galactic: ["l","b"],
	c.ICRS: ["ra","dec"]}
