# This module contains functions used to implement the motion-compensated coordinate
# system used for the Planet 9 search. We use mjd throughout to be consistent with
# the coordinates module.

import numpy as np, ephem
from enlib import utils, parallax, ephemeris, coordinates, fft
from scipy import interpolate

def smooth(arr, n):
	arr = np.array(arr)
	fa  = fft.rfft(arr)
	fa[n:] = 0
	fft.irfft(fa, arr, normalize=True)
	return arr

class MotionCompensator:
	def __init__(self, obj=None, nsamp=1000, nsmooth=20):
		if obj is None:
			obj = ephemeris.make_object(a=500, e=0.25, inc=20, Omega=90, omega=150)
		self.o    = obj
		self.tref = ephemeris.djd2mjd(obj._epoch)
		# Evaluate a coarse, smooth orbit.
		dt   = self.period / nsamp
		t    = self.tref + (np.arange(0, nsamp)-nsamp//2)*dt
		dt   = (t[-1]-t[0])/(len(t)-1)
		pos  = ephemeris.ephem_pos(obj, t)
		r    = pos[2]
		pos, sundist  = parallax.earth2sun(pos[:2], r, t)
		# Use two points 1/4 of an orbit apart to find the spin vector of the orbit
		refi   = [0,len(t)/4]
		v1, v2 = utils.ang2rect(pos[:,refi], zenith=False).T
		vzen   = np.cross(v1,v2)
		self.pzen = utils.rect2ang(vzen, zenith=False)
		# Transform the orbit to orbit-aligned coordinates so that the rotation is purely
		# in the lon direction
		pos_oo = coordinates.recenter(pos, self.pzen)
		# Smooth the orbit to get rid of residual parallax that wasn't fully
		# corrected due to pyephem adding light travel delay
		lon     = utils.unwind(np.arctan2(
					smooth(np.sin(pos_oo[0]),nsmooth),
					smooth(np.cos(pos_oo[0]),nsmooth)
				))
		sundist = smooth(sundist, nsmooth)
		self.lon0   = lon[0]
		x           = lon-self.lon0
		# Build our interpolators. We need lon(x), dist(x) and t(x)
		time_spline = interpolate.splrep(x, t)
		speed       = 1/interpolate.splev(x, time_spline, der=1)
		self.speed_spline = interpolate.splrep(x, speed)
		self.dist_spline  = interpolate.splrep(x, sundist)
	@property
	def period(self):
		return utils.yr * self.o._a**1.5
	def compensate(self, pos, t, tref):
		"""Approximately compensate for how much an object currently
		at position pos has moved since the reference time tref, assuming
		it has a similar orbit to us. Compensates for both orbital motion
		and parallax."""

		# Forwards: ref pos -> orbit correction -> parallax displacement -> obs pos
		# Backwards: obs pos -> anti-parallax -> anti-orbit -> ref pos
		# But parallax depends on distance, which depends on where in the orbit
		# one is, but to get that we should be parallax-corrected so we don't
		# think that the parallax motion represents being further along the orbit.
		# So we should ideally iterate the parallax correction.
		#
		# Once we're parallax corrected we want anti-orbit. This should be straightforward at
		# this point


		# Transform to orbit-aligned coordinates, get the distance, and use
		# it to apply parallax correction
		pos_oo = coordinates.recenter(pos, self.pzen)
		x       = (pos_oo[0]-self.lon0)%(2*np.pi)
		sundist = interpolate.splev(x, self.dist_spline)
		pos_sunrel, earthdist = parallax.earth2sun_mixed(pos, sundist, t)
		# Then apply the orbital correction
		pos_oo  = coordinates.recenter(pos_sunrel, self.pzen)
		x       = (pos_oo[0]-self.lon0)%(2*np.pi)
		# We will use a 2nd order orbital correction, so we need
		# speed and dspeed/dt = dspeed/dx * dx/dt = speed * dspeed
		speed   = interpolate.splev(x, self.speed_spline)
		accel   = interpolate.splev(x, self.speed_spline, der=1) * speed
		# Apply orbit correction
		delta_t = t-tref
		pos_oo[0] -= delta_t*speed + delta_t**2*accel
		# Transform back to celestial coordinates
		opos = coordinates.decenter(pos_oo, self.pzen)
		return opos
