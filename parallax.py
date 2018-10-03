"""This module provides functions for dealing with apparent parallax displacement
in maps. This module works in equatorial coordinates."""
import numpy as np
from . import utils, ephemeris

def sun2earth(pos_sunrel, sundist, time, mul=1):
	"""Apply parallax to the given sun_relative coordinates pos_sunrel[{ra,dec},...]
	in radians and corresponding distances sundist in AU, for the given times in MJD.
	Returns earth_relative coordinates and distances as a tuple."""
	# Compute the position of the object relative to the earth at each time
	vec_obj_sunrel   = utils.ang2rect(pos_sunrel, zenith=False)*sundist
	vec_sun_earthrel = ephemeris.ephem_vec("Sun", time)
	vec_obj_earthrel   = (vec_obj_sunrel.T + mul*vec_sun_earthrel.T).T
	# Translate this into an angular position
	pos_earthrel = utils.rect2ang(vec_obj_earthrel, zenith=False)
	earthdist = np.sum(vec_obj_earthrel**2,0)**0.5
	return pos_earthrel, earthdist

def earth2sun(pos_earthrel, earthdist, time):
	"""Remove parallax to the given earth-relative coordinates pos_earthrel[{ra,dec},...]
	in radians and corresponding distances earthdist in AU, for the given times in MJD.
	Returns sun-relative coordinates and distances as a tuple."""
	return sun2earth(pos_earthrel, earthdist, time, -1)

def earth2sun_mixed(pos_earthrel, sundist, time):
	"""Like earth2sun, but distances are sun-relative instead of earth-relative.
	Returns pos_sunrel, earthdist. FIXME: For planets interior to Earth, this
	function may give the wrong result.
	"""
	# Compute the position of the object relative to the earth at each time
	vec_obj_earthrel = utils.ang2rect(pos_earthrel, zenith=False)
	vec_earth_sunrel = -ephemeris.ephem_vec("Sun", time)
	# Get the object earth distance based on the sun distance, using
	# the law of sines: sin(obj_earth_sun)/obj_sun = sin(sun_obj_earth)/obj_earth.
	b = np.sum(vec_earth_sunrel**2,0)**0.5
	c = sundist
	cosC = np.sum(-(vec_obj_earthrel.T*vec_earth_sunrel.T).T,0)/b
	C    = np.arccos(cosC)
	sinB = np.sin(C) * b/c
	# This assumes that B < 90 degrees. For interior planets this can
	# break down. In this case there may not be an unambiguous solution.
	sinA = np.sin(np.pi - np.arcsin(sinB) - C)
	a = b * sinA/sinB
	earthdist = a
	vec_obj_earthrel *= earthdist
	# Get the relative position
	vec_obj_sunrel = (vec_obj_earthrel.T + vec_earth_sunrel.T).T
	# Translate this into an angular position
	pos_sunrel = utils.rect2ang(vec_obj_sunrel, zenith=False)
	return pos_sunrel, earthdist
