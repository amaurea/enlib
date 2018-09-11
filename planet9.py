# This module contains functions used to implement the motion-compensated coordinate
# system used for the Planet 9 search. We use mjd throughout to be consistent with
# the coordinates module.

import numpy as np, ephem
from enlib import utils, parallax, ephemeris, coordinates, fft
from scipy import interpolate, special

def smooth(arr, n):
	arr = np.array(arr)
	fa  = fft.rfft(arr)
	fa[n:] = 0
	fft.irfft(fa, arr, normalize=True)
	return arr

def fourier_deriv(arr, order, dt=1):
	# Our fourier convention is
	# da/dx = d/dx sum exp(2 pi 1j x f/N) fa(f) = sum 2 pi 1j f/N exp(2 pi 1j x f/N) fa(f)
	# = ifft(2 pi 1j f/N fa)
	arr = np.array(arr)
	n   = arr.shape[-1]
	fa  = fft.rfft(arr)
	f   = fft.rfftfreq(n, dt)
	fa *= (2j*np.pi*f)**order
	fft.irfft(fa, arr, normalize=True)
	return arr

def orb_subsample(t, step=0.1, nsub=3):
	offs = (np.arange(nsub)-(nsub-1)/2.0)*step
	return (t[:,None] + offs[None,:]).reshape(-1)

# Notes:
#
# 1. The small earth parallax residual results in a small discontiunity
#    at the wraparound, which leads an extreme value for the speed at the
#    edge. This leads to spline problems. The discontinuity also leads to
#    ringing when smoothing. Could deal with this by padding beyond one orbit
# 2. It's really annoying that this should be so hard. Maybe it's easier to
#    just skip pyephem and compute the ideal orbit myself.
# 3. The main question we're asking here is:
#     if the object was at position p at time t1, where was it at time t0?
#    Is it possible to answer this question directly with pyephem, without
#    any manual parallax work?

class MotionCompensator:
	def __init__(self, obj=None, nsamp=1000, nderiv=2):
		if obj is None:
			obj = ephemeris.make_object(a=500, e=0.25, inc=20, Omega=90, omega=150)
		self.o    = obj
		self.tref = ephemeris.djd2mjd(obj._epoch)
		# Evaluate a coarse, smooth orbit.
		dt   = self.period / nsamp
		t    = self.tref + (np.arange(nsamp)-nsamp//2)*dt
		pos  = ephemeris.trace_orbit(obj, t, self.tref, nsub=20)
		# Find the normal vector for the orbit. We use a lot of cross products to make it
		# robust to deviations from ellipticity
		vs    = utils.ang2rect(pos[:2], zenith=False)
		vzen  = np.mean(np.cross(vs, np.roll(vs,-1,1),0,0),0)
		self.pzen = utils.rect2ang(vzen, zenith=False)
		# Transform the orbit to orbit-aligned coordinates so that the rotation is purely
		# in the lon direction
		pos_oo = utils.unwind(coordinates.recenter(pos[:2], self.pzen))
		# Now use the sub-points to compute the lon, dlon/dt and ddlon/dtdt
		lon     = pos_oo[0]
		sundist = pos[2,:]
		# Hack: smooth lon and sundist to avoid pyephem-induced jitter that
		# especially affects the speed and accel. We have to extract the average
		# orbital motion before smoothing, to avoid discontinuity
		nkeep     = 20
		avg_speed = 2*np.pi/self.period
		delta_lon = lon - avg_speed * np.arange(nsamp)*dt
		delta_lon = smooth(delta_lon, nkeep)
		#sundist   = smooth(sundist,   nkeep)
		lon       = delta_lon + avg_speed * np.arange(nsamp)*dt
		# Compute and spline the derivatives
		self.lon0   = lon[0]
		x           = utils.unwind(lon-self.lon0)
		self.dist_spline   = interpolate.splrep(x, sundist)
		self.deriv_splines = [interpolate.splrep(x, fourier_deriv(delta_lon, i+1, dt=dt) + (avg_speed if i == 0 else 0)) for i in range(nderiv)]
		# Debug
		self.lon = lon
		#self.speed = speed
		#self.accel = accel
		self.sundist = sundist
		self.pos_oo = pos_oo
		self.t = t
		self.dt = dt
		self.pos = pos
	@property
	def period(self):
		return ephemeris.yr * self.o._a**1.5
	def compensate(self, pos, t, tref, nit=2):
		"""Approximately compensate for how much an object currently
		at position pos has moved since the reference time tref, assuming
		it has a similar orbit to us. Compensates for both orbital motion
		and parallax."""
		# First find the sun distance and sun-relative coordinates
		pos_sunrel = pos
		for i in range(nit):
			pos_oo  = coordinates.recenter(pos_sunrel, self.pzen)
			x       = (pos_oo[0]-self.lon0)%(2*np.pi)
			sundist = interpolate.splev(x, self.dist_spline)
			pos_sunrel, earthdist = parallax.earth2sun_mixed(pos, sundist, t)
		# Then apply the orbital correction
		pos_oo  = coordinates.recenter(pos_sunrel, self.pzen)
		x       = (pos_oo[0]-self.lon0)%(2*np.pi)
		delta_t = t-tref
		old = pos_oo.copy()
		for i, spline in enumerate(self.deriv_splines):
			deriv = interpolate.splev(x, spline)
			pos_oo[0] -= delta_t**(i+1)/special.factorial(i+1) * deriv
		# Transform back to celestial coordinates
		opos = coordinates.decenter(pos_oo, self.pzen)
		return opos
