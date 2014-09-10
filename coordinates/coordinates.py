"""This module provides conversions between astronomical coordinate systems.
When c is more developed, it might completely replace this
module. For now, it is used as a part of the implementation."""
import numpy as np, pyfsla
import astropy.coordinates as c, astropy.units as u, ephem
from enlib import iers, utils
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
	if from_ref != None: coords = decenter(coords, from_ref, unit)
	if from_sys != to_sys:
		if from_sys == c.AltAz:
			if unit != u.rad: coords = coords * unit.in_units(u.rad)
			coords = hor2cel(coords, time, site)
			if unit != u.rad: coords = coords / unit.in_units(u.rad)
		coords = transform_astropy(nohor(from_sys), nohor(to_sys), coords, unit)
		if to_sys == c.AltAz:
			if unit != u.rad: coords = coords * unit.in_units(u.rad)
			coords = cel2hor(coords, time, site)
			if unit != u.rad: coords = coords / unit.in_units(u.rad)
	if to_ref != None: coords = recenter(coords, to_ref, unit)
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

def hor2cel(coord, time, site):
	coord  = np.asarray(coord)
	info   = iers.lookup(time[0])
	as2rad = np.pi/180/60/60
	ao = slalib.sla_aoppa(time[0], info.dUT, site.lon*np.pi/180, site.lat*np.pi/180, site.alt,
		info.pmx*as2rad, info.pmy*as2rad, site.T, site.P, site.hum,
		299792.458/site.freq, 0.0065)
	am = slalib.sla_mappa(2000.0, time[0])
	# This involves a transpose operation, which is not optimal
	return pyfsla.aomulti(time, coord, ao, am)

def cel2hor(coord, time, site):
	coord  = np.asarray(coord)
	info   = iers.lookup(time[0])
	as2rad = np.pi/180/60/60
	ao = slalib.sla_aoppa(time[0], info.dUT, site.lon*np.pi/180, site.lat*np.pi/180, site.alt,
		info.pmx*as2rad, info.pmy*as2rad, site.T, site.P, site.hum,
		299792.458/site.freq, 0.0065)
	am = slalib.sla_mappa(2000.0, time[0])
	# This involves a transpose operation, which is not optimal
	return pyfsla.oamulti(time, coord, ao, am)

def rotmatrix(ang, axis, unit):
	unit = getunit(unit)
	axis = axis.lower()
	if unit != u.rad: ang = ang * unit.in_units(u.rad)
	c, s = np.cos(ang), np.sin(ang)
	if axis == "x": return np.array([[ 1, 0, 0],[ 0, c,-s],[ 0, s, c]])
	if axis == "y": return np.array([[ c, 0, s],[ 0, 1, 0],[-s, 0, c]])
	if axis == "z": return np.array([[ c,-s, 0],[ s, c, 0],[ 0, 0, 1]])
	raise ValueError("Axis %s not recognized" % axis)

def euler_mat(euler_angles, kind="zyz", unit="rad"):
	"""Defines the rotation matrix M for a ABC euler rotation,
	such that M = A(alpha)B(beta)C(gamma), where euler_angles =
	[alpha,beta,gamma]. The default kind is ABC=ZYZ."""
	unit = getunit(unit)
	alpha, beta, gamma = euler_angles
	R1 = rotmatrix(gamma, kind[2], unit)
	R2 = rotmatrix(beta,  kind[1], unit)
	R3 = rotmatrix(alpha, kind[0], unit)
	return R3.dot(R2).dot(R1)

# Why do I have to define these myself?
def ang2rect(angs, zenith=True):
	phi, theta = angs
	ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
	if zenith: return np.array([st*cp,st*sp,ct])
	else:      return np.array([ct*cp,ct*sp,st])
def rect2ang(rect, zenith=True):
	x,y,z = rect
	r     = (x**2+y**2)**0.5
	phi   = np.arctan2(y,x)
	if zenith: theta = np.arctan2(r,z)
	else:      theta = np.arctan2(z,r)
	return np.array([phi,theta])

def euler_rot(euler_angles, coords, kind="zyz", unit="rad"):
	unit   = getunit(unit)
	coords = np.asarray(coords)
	co     = coords.reshape(2,-1)
	if unit != u.rad: co = co * unit.in_units(u.rad)
	M      = euler_mat(euler_angles, kind, unit)
	rect   = ang2rect(co, False)
	rect   = M.dot(rect)
	co     = rect2ang(rect, False)
	if unit != u.rad: co = co / unit.in_units(u.rad)
	return co.reshape(coords.shape)

def recenter(angs, center, unit="rad"):
	"""Recenter coordinates "angs" (as ra,dec) on the location given by "center",
	such that center moves to the north pole."""
	unit = getunit(unit)
	ra0, dec0 = center
	return euler_rot([ra0,dec0-90/unit.in_units(u.deg),-ra0], angs, kind="zyz", unit=unit)
def decenter(angs, center, unit="rad"):
	"""Inverse operation of recenter."""
	unit = getunit(unit)
	ra0, dec0 = center
	return euler_rot([ra0,90/unit.in_units(u.deg)-dec0,-ra0],  angs, kind="zyz", unit=unit)

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
	refsys = getsys(refsys) if refsys != None else base
	if ref is None: return [base, ref]
	if isinstance(ref, basestring):
		# In our first format, ref is a set of coordinates in degrees
		try:
			ref = np.asfarray(ref.split(","))
			assert(ref.ndim == 1 and len(ref) == 2)
		except ValueError:
			# Otherwise, treat as an ephemeris object
			ref    = ephem_pos(ref, time)/unit.in_units(u.rad)
			refsys = getsys("equ")
	# Now rotate the reference point to our base system
	if refsys != None:
		ref = transform(refsys, base, ref, unit=unit, time=time, site=site)
	return [base, ref]

def ephem_pos(name, mjd):
	"""Given the name of an ephemeris object from pyephem and a
	time in modified julian date, return its position in ra, dec
	in radians in equatorial coordinates."""
	djd = mjd + 2400000.5 - 2415020
	obj = getattr(ephem, name)()
	obj.compute(djd)
	return np.array([float(obj.ra), float(obj.dec)])

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
