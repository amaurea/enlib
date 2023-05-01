"""Helper functions for pyephem."""
from __future__ import division, print_function
import numpy as np, ephem, os
from . import utils
try: basestring
except NameError: basestring = str

yr = 365.2425
def mjd2djd(mjd): return np.asarray(mjd) + 2400000.5 - 2415020
def djd2mjd(djd): return np.asarray(djd) - 2400000.5 + 2415020

def define_subsamples(t, dt=10):
	t = np.asarray(t)
	if t.ndim == 0: return np.array([t]), np.array([0])
	if dt == 0: return t, np.arange(len(t))
	box       = utils.widen_box([np.min(t),np.max(t)], 1e-2, relative=True)
	sub_nsamp = max(3,int((box[1]-box[0])/dt))
	if sub_nsamp > len(t): return t, np.arange(len(t))
	sub_t     = np.linspace(box[0], box[1], sub_nsamp, endpoint=True)
	return sub_t, (t-box[0])*(sub_nsamp-1)/(box[1]-box[0])

def get_object(objname):
	if isinstance(objname, basestring):
		return getattr(ephem, objname)()
	else:
		return objname

def ephem_raw(objname, mjd, lat=None, lon=None, alt=None, nodelay=False):
	"""Simple wrapper around pyephem. Returns astrometric ra, dec, rad (AU)
	for each specified modified julian date for the given object name
	(case sensitive)."""
	mjd = np.asarray(mjd)
	if nodelay:
		# Undo pyephem's light travel time stuff. This is not properly tested
		dist  = ephem_raw(objname, mjd, nodelay=False)[2]
		delay = dist*utils.AU/utils.c/utils.day2sec
		return ephem_raw(objname, mjd - delay, nodelay=False)
	djd = mjd2djd(mjd)
	obj = get_object(objname)
	res = np.zeros([3,len(djd)])
	# Full observer if available
	if lon is not None:
		site = ephem.Observer()
		site.lon = lon
		site.lat = lat
		site.elevation = alt
	for I in utils.nditer(djd.shape):
		if lon is not None:
			site.date = djd[I]
			obj.compute(site)
		else:
			obj.compute(djd[I])
		res[0,I] = obj.a_ra
		res[1,I] = obj.a_dec
		try: res[2,I] = obj.earth_distance
		except AttributeError: res[2,I] = 1.0
	res.reshape((3,)+mjd.shape)
	return res

def ephem_vec(objname, mjd, dt=10, lat=None, lon=None, alt=None, nodelay=False):
	"""Computes the earth-relative position vector[{x,y,z},ntime] for the
	given object. Uses interpolation in steps dt (seconds) to speed things up.
	Set dt to 0 to disable this. The resulting vector has units of AU."""
	# Get low-res [ra,dec,r]
	sub_time, inds = define_subsamples(mjd, dt=dt/(24.*3600))
	sub_pos  = ephem_raw(objname, sub_time, lat=lat, lon=lon, alt=alt, nodelay=nodelay)
	# Convert to low-res [x,y,z]
	sub_vec  = utils.ang2rect(sub_pos[:2],zenith=False)*sub_pos[2]
	# Interpolate to target resolution
	full_vec = utils.interpol(sub_vec, inds[None])
	return full_vec

def ephem_pos(objname, mjd, dt=10, lat=None, lon=None, alt=None, nodelay=False):
	"""Computes the earth-relative angular position and radius [{ra,dec,r},ntime]
	for the given object. Uses interpolation in steps dt (seconds) to sepeed
	things up. Set dt to 0 to disable this. r has units of AU."""
	# Get low-res [ra,dec,r]
	sub_time, inds = define_subsamples(mjd, dt=dt/(24.*3600))
	sub_pos = ephem_raw(objname, sub_time, lat=lat, lon=lon, alt=alt, nodelay=nodelay)
	# Avoid angle wraps, as they will mess up the interpolation
	sub_pos[0] = utils.unwind(sub_pos[0])
	# Interpolate to target resolution
	full_pos = utils.interpol(sub_pos, inds[None])
	return full_pos

def trace_orbit(obj, mjd, tref, nsub=1, Psub=yr):
	"""Trace a constant-time orbit, showing where the object would go
	if nothing else changed other than its position along the orbit (e.g.
	parallax, precession etc. held constant. The orbit is evaluated at
	the times given by mjd. Since everything else is held fixed, this
	is just a proxy for the mean anomaly, such that
	M = M0 + 2*pi*(mjd-tref)/P, where M0 is the standard mean anomaly
	from the orbital elements and P is the time at which everything else
	was fixed.
	
	Nsub specifies the number of subsamples per earth orbit at which
	the trace should be repeated. The purpose of this is to average out
	the effect of parallax. nsub=2 should be sufficient for this. nsub=1
	(the default) disables this subsampling, resulting in an orbit trace
	that includes the effect of whatever position the Earth happened to
	be at at tref.

	Unlike what happens when normally evaluating the orbit for one period,
	this trace is guaranteed to be properly periodic.
	"""
	mjd  = np.asarray(mjd)
	if nsub > 1:
		# Subsample the yearly parallax
		res  = np.zeros((3,)+mjd.shape)
		vec  = np.zeros((3,)+mjd.shape)
		offs = (np.arange(nsub)-(nsub-1)/2.0)*Psub/nsub
		for off in offs:
			pos  = trace_orbit(obj, mjd-off, tref+off)
			vec += utils.ang2rect(pos[:2],zenith=False)*pos[2]
		vec /= nsub
		res[:2] = utils.rect2ang(vec,zenith=False)
		res[2]  = np.sum(vec**2,0)**0.5
		return res
	obj  = obj.copy()
	M0   = obj._M
	P    = obj._a**1.5*yr
	with utils.flatview(mjd, mode="r") as tflat:
		djd = mjd2djd(tref)
		obj = get_object(obj)
		res = np.zeros([3,len(tflat)])
		for i, t in enumerate(tflat):
			# ._M is radians when read but degrees when set, so must compensate
			obj._M = (M0 + 2*np.pi*(t-tref)/P)/utils.degree
			obj.compute(djd)
			res[0,i] = obj.a_ra
			res[1,i] = obj.a_dec
			res[2,i] = obj.earth_distance
	res.reshape((3,)+mjd.shape)
	return res

def make_object(a=1, e=0, inc=0, M=0, Omega=0, omega=0, epoch=36525, epoch_M=36525):
	"""Construct a pyephem EllipticalBody with the given parameters."""
	o = ephem.EllipticalBody()
	o._a   = a
	o._e   = e
	o._inc = inc
	o._Om  = Omega
	o._om  = omega
	o._M   = M
	o._epoch = epoch
	o._epoch_M = epoch_M
	return o

def read_object(fname):
	try:
		return read_object_xephem(fname)
	except (IOError, OSError, ValueError) as e:
		return read_object_simple(fname)

def read_object_xephem(fname):
	with open(fname, "r") as f:
		for line in f:
			if line.startswith("#"): continue
			return ephem.readdb(line)
	raise IOError("No xephem data in '%s'" % fname)

def read_object_simple(fname):
	"""Construct a pyephem EllipticalBody based on the key = value entries in
	the given parameter file."""
	o = make_object()
	o._epoch_M = 0
	fields = [("a","_a"),("e","_e"),("inc","_inc"),("M","_M"),("Omega","_Om"),("omega","_om"),("epoch","_epoch"),("epoch_M","_epoch_M")]
	with open(fname, "r") as f:
		for line in f:
			line = line.split("#")[0]
			toks = line.split()
			if len(toks) == 0: continue
			if len(toks) != 3 or toks[1] != "=":
				raise IOError("Format error in '%s'" % fname)
			for fname, oname in fields:
				if toks[0] == fname:
					setattr(o, oname, float(toks[2]))
	if not o._epoch_M: o._epoch_M = o._epoch
	return o

def register_object(name, obj):
	"""Register the given object in pyephem, so it appears like a predefined object"""
	setattr(ephem, name, lambda: obj)

# Asteroid stuff

def phase_offset(mjd, pos, dist, lat=0, lon=0, alt=0):
	"""Given an asteroid or other object's position pos[{ra,dec},...] and
	distance from earth dist[...] in AU at times mjd[...], compute the
	offset between the object's phase angle relative to the stars and it's
	phase angle relative to the Earth, assuming a planar solar system.

	Returns (phase_offset, time_delay). With this,
	phase_obs = phase_raw + phase_offset - 2*np.pi*time_delay/period,
	where phase_obs is the rotation angle we see from the earth, phase_raw
	is the rotation angle relative to a fixed direction in space,
	time_delay is the time light takes to travel from the asteroid to the
	Earth in seconds, and period is the asteroid's rotation period in seconds.
	
	The apparent phase has nothing to do with illumination from the sun. It simply
	describes which part of the asteroid faces us.

	NB! We assume that all orbits in the solar system occur in the ecliptic. This
	means that the asteroid spin can be described simply by its period. However,
	this still leaves the freedom to spin in the positive (right-handed) and
	negative directions. This means that when calculating the raw phase you
	pass into this function, and when applying the time delay correction,
	you should make sure period has the right sign!
	"""
	# Broadcast and flatten inputs
	from . import coordinates
	mjd, ra, dec, dist = np.broadcast_arrays(mjd, pos[0], pos[1], dist)
	shape = mjd.shape
	mjd, ra, dec, dist = [a.reshape(-1) for a in [mjd, ra, dec, dist]]
	# Get the sun's position at the same times
	ra_sun, dec_sun, dist_sun = ephem_raw("Sun", mjd, lat=lat, lon=lon, alt=alt)
	# At this point we have everything in Site-relative equatorial coordinates.
	# But we want the asteroid and Earth in Sun-relative ecliptic coordinates.
	# Start by transforming to Sun-relative coordinates
	# Earth-to-asteroid vector and Earth-to-sun vector
	v_ea    = utils.ang2rect([ra,     dec    ])*dist
	v_es    = utils.ang2rect([ra_sun, dec_sun])*dist_sun
	# Sun-to-earth vector and Sun-to-asteroid vectors
	v_se    = -v_es
	v_sa    = v_se + v_ea
	# Distances
	r_ea = dist
	r_es = dist_sun
	r_sa = np.sum(v_sa**2,0)**0.5
	# Back to coordinates, which will now be sun-centered equatorial coordinates
	sequ_earth = utils.rect2ang(v_se)
	sequ_ast   = utils.rect2ang(v_sa)
	# To ecliptic coordinates
	ecl_earth = coordinates.transform("equ", "ecl", sequ_earth)
	ecl_ast   = coordinates.transform("equ", "ecl", sequ_ast)
	# Solve for the sun-object-earth angle alpha
	delta_lon = ecl_ast[0] - ecl_earth[0]
	sin_alpha = np.sin(delta_lon) * r_es/r_ea
	cos_alpha = (r_sa**2 + r_ea**2 - r_es**2)/(2*r_sa*r_ea)
	alpha     = np.arctan2(sin_alpha, cos_alpha)
	phase_offset = -alpha - ecl_ast[0] + np.pi
	# We will also need to handle light travel delay. These has two parts:
	# 1. The effect on the Earth and asteroid's coordinates. This is tiny,
	#    so we don't worry about it. It might already be taken into account,
	#    but would need to think more about this. But anyway, it doesn't matter.
	# 2. We see the asteroid's rotation in the past. We can't correct for this
	#    without knowing the asteroid's rotation period, which isn't a part of
	#    this functions arguments. Could make the user pass in the period, but
	#    more flexible to just return the time delay too.
	time_delay = dist * utils.AU/utils.c
	# Reshape to target shape
	phase_offset, time_delay = [a.reshape(shape) for a in [phase_offset, time_delay]]
	return phase_offset, time_delay
