"""This module provides conversions between astronomical coordinate systems.
When c is more developed, it might completely replace this
module. For now, it is used as a part of the implementation."""
import numpy as np, pyfsla, iers
import astropy.coordinates as c, astropy.units as u
from pyslalib import slalib

def transform(from_sys, to_sys, coords, unit="radians", time=None, site=None):
	"""Transforms coords[2,N] from system from_sys to system to_sys, where
	systems can be "hor", "cel" or "gal". For transformations involving
	"hor", the optional arguments time (in modified julian days) and site (which must
	contain .lat (rad), .lon (rad), .P (pressure, mBar), .T (temperature, K),
	.hum (humidity, 0.2 by default), .alt (altitude, m). Returns an array
	with the same shape as the input."""
	# Make ourselves case insensitive, and look up the corresponding objects
	unit, from_sys, to_sys = getunit(unit), getsys(from_sys), getsys(to_sys)
	if from_sys == to_sys: return coords
	if from_sys == c.AltAz:
		if unit != u.rad: coords = coords * unit.in_units(u.rad)
		coords = hor2cel(coords, time, site)
		if unit != u.rad: coords = coords / unit.in_units(u.rad)
	coords = transform_astropy(nohor(from_sys), nohor(to_sys), coords, unit)
	if to_sys == c.AltAz:
		if unit != u.rad: coords = coords * unit.in_units(u.rad)
		coords = cel2hor(coords, time, site)
		if unit != u.rad: coords = coords / unit.in_units(u.rad)
	return coords

def transform_astropy(from_sys, to_sys, coords, unit):
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
	ao = slalib.sla_aoppa(time[0], info.dUT, site.lon, site.lat, site.alt,
		info.pmx*as2rad, info.pmy*as2rad, site.T, site.P, site.hum,
		299792.458/site.freq, 0.0065)
	am = slalib.sla_mappa(2000.0, time[0])
	# This involves a transpose operation, which is not optimal
	return pyfsla.aomulti(time, coord, ao, am)

def cel2hor(coord, time, site):
	coord  = np.asarray(coord)
	info   = iers.lookup(time[0])
	as2rad = np.pi/180/60/60
	ao = slalib.sla_aoppa(time[0], info.dUT, site.lon, site.lat, site.alt,
		info.pmx*as2rad, info.pmy*as2rad, site.T, site.P, site.hum,
		299792.458/site.freq, 0.0065)
	am = slalib.sla_mappa(2000.0, time[0])
	# This involves a transpose operation, which is not optimal
	return pyfsla.oamulti(time, coord, ao, am)

def nohor(sys): return sys if sys != c.AltAz else c.ICRS
def getsys(sys): return str2sys[sys.lower()] if isinstance(sys,basestring) else sys
def getunit(u): return str2unit[u.lower()] if isinstance(u,basestring) else u

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
