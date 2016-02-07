"""This module provides functions for dealing with apparent parallax displacement
in maps. This module works in equatorial coordinates."""
import numpy as np
from enlib import utils, ephemeris

def apparent_displacement(pos, time, sundist):
	"""Compute the apparent displacement pos_obs - pos_mean for
	a source with mean position given by pos[{ra,dec},:] at the given time,
	and with the given distance from the sun (in AU)."""
	# Compute the position of the object relative to the earth at each time
	vec_obj_sun   = utils.ang2rect(pos, zenith=False)*sundist
	vec_sun_earth = ephemeris.ephem_vec("Sun", time)
	vec_obj_earth = vec_obj_sun - vec_sun_earth
	# Translate this into an angular position
	pos_obs = utils.rect2ang(vec_obj_earth, zenith=False)
	# And turn that into a displacement
	offset = pos_obs - pos
	offset[0] = utils.rewind(offset[0])
	return offset
