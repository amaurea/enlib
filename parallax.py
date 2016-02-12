"""This module provides functions for dealing with apparent parallax displacement
in maps. This module works in equatorial coordinates."""
import numpy as np
from enlib import utils, ephemeris

def sun2earth(pos_sunrel, time, sundist, diff=False):
	"""Compute the apparent displacement pos_earthrel - pos_sunrel for
	a source with sun-relative pointing given by pos_sunrel[{ra,dec},:] at the given time,
	and with the given distance from the sun (in AU)."""
	# Compute the position of the object relative to the earth at each time
	vec_obj_sunrel   = utils.ang2rect(pos_sunrel, zenith=False)*sundist
	vec_sun_earthrel = ephemeris.ephem_vec("Sun", time)
	vec_obj_earthrel = (vec_obj_sunrel.T + vec_sun_earthrel.T).T
	# Translate this into an angular position
	pos_earthrel = utils.rect2ang(vec_obj_earthrel, zenith=False)
	if not diff: return pos_earthrel
	# And turn that into a displacement
	offset = pos_earthrel - pos_sunrel
	offset[1] = utils.rewind(offset[1])
	return offset

def earth2sun(pos_earthrel, time, sundist, diff=False):
	"""Compute the apparent displacement pos_sunrel - pos_earthrel for
	a source with earth-relative pointing given by pos_earthrel[{ra,dec},:] at the given time,
	and with the given distance from the sun (in AU)."""
	# Compute the position of the object relative to the earth at each time
	vec_obj_earthrel = utils.ang2rect(pos_earthrel, zenith=False)
	vec_earth_sunrel = -ephemeris.ephem_vec("Sun", time)
	# Get the object earth distance based on the sun distance, using
	# the law of sines: sin(obj_earth_sun)/obj_sun = sin(sun_obj_earth)/obj_earth.
	b = np.sum(vec_earth_sunrel**2,0)**0.5
	c = sundist
	cosC = np.sum(-(vec_obj_earthrel.T*vec_earth_sunrel.T).T,0)/b
	sinC = (1-cosC**2)**0.5
	sinB = sinC * b / c
	sinA = np.sin(np.pi - np.arcsin(sinB) - np.arcsin(sinC))
	a = b * sinA/sinB
	earthdist = a
	vec_obj_earthrel *= earthdist
	# Get the relative position
	vec_obj_sunrel = (vec_obj_earthrel.T + vec_earth_sunrel.T).T
	# Translate this into an angular position
	pos_sunrel = utils.rect2ang(vec_obj_sunrel, zenith=False)
	if not diff: return pos_sunrel
	# And turn that into a displacement
	offset = pos_sunrel - pos_earthrel
	offset[1] = utils.rewind(offset[1])
	return offset
