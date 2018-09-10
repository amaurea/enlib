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
	def __init__(self, obj=None, nsamp=1000):
		if obj is None:
			obj = ephemeris.make_object(a=500, e=0.25, inc=20, Omega=90, omega=150)
		self.o    = obj
		self.tref = ephemeris.djd2mjd(obj._epoch)
		# Evaluate a coarse, smooth orbit.
		dt   = self.period / nsamp
		t    = self.tref + (np.arange(nsamp)-nsamp//2)*dt
		pos  = ephemeris.trace_orbit(obj, t, self.tref, nsub=20)
		# Use two points 1/4 of an orbit apart to find the spin vector of the orbit
		refi   = [0,len(t)/4]
		v1, v2 = utils.ang2rect(pos[:2,refi], zenith=False).T
		vzen   = np.cross(v1,v2)
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
		nkeep   = 20
		avg_speed = 2*np.pi/self.period
		delta_lon = lon - avg_speed * np.arange(nsamp)*dt
		self.delta_lon = delta_lon
		delta_lon = smooth(delta_lon, nkeep)
		#sundist   = smooth(sundist,   nkeep)
		lon       = delta_lon + avg_speed * np.arange(nsamp)*dt
		# Compute the speed and acceleration
		speed   = fourier_deriv(delta_lon, 1, dt=dt) + avg_speed
		accel   = fourier_deriv(delta_lon, 2, dt=dt)
		# Build our interpolators
		self.lon0   = lon[0]
		x           = utils.unwind(lon-self.lon0)
		self.speed_spline = interpolate.splrep(x, speed)
		self.accel_spline = interpolate.splrep(x, accel)
		self.dist_spline  = interpolate.splrep(x, sundist)
		# Debug
		self.lon = lon
		self.speed = speed
		self.accel = accel
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
		speed   = interpolate.splev(x, self.speed_spline)
		accel   = interpolate.splev(x, self.accel_spline)
		delta_t = t-tref
		old = pos_oo.copy()
		pos_oo[0] -= delta_t*speed + delta_t**2*accel
		# Transform back to celestial coordinates
		opos = coordinates.decenter(pos_oo, self.pzen)
		return opos

#def solve_kepler(M, e, n=10):
#	E = M if e < 0.8 else np.pi
#	for i in range(n):
#		E -= (E-e*np.sin(E)-M)/(1-e*np.cos(E))
#	return E
#
#class SimpleEllipse:
#	def __init__(self, a=1, e=0, inc=0, Om=0, om=0, M=0, epoch=36525, eps=23.43686):
#		self.a, self.e, self.inc, self.Om, self.om, self.M, self.epoch, self.eps = a, e, inc, Om, om, M, epoch, eps
#		rot_orient  = coordinates.euler_mat([Om*utils.degree, inc*utils.degree, om*utils.degree])
#		rot_ecl2cel = utils.rotmatrix(eps*utils.degree, "x")
#		self.R = rot_ecl2cel.dot(rot_orient)
#	@property
#	def P(self): return self.a**1.5*ephemeris.yr
#	@property
#	def b(self): return self.a * (1-self.e**2)**0.5
#	def calc_rect(self, time):
#		time = np.asarray(time)
#		M = self.M + 2*np.pi*(time - self.epoch)/self.P
#		E = solve_kepler(M, self.e)
#		p = np.zeros((3,) + time.shape)
#		p[0] = self.a * (np.cos(E)-self.e)
#		p[1] = self.b * np.sin(E)
#		return np.einsum("ab,b...->a...", self.R, p)
#	def calc_ang(self, time):
#		p = self.calc_rect(time)
#		return utils.rect2ang(p, zenith=False)
