# This module handles precise timing for pulsars with timing tables available,
# like the crab pulsar.
import numpy as np
from . import utils, bunch, coordinates, iers

# Main interface

def obstime2phase(ctime, target, pulsar_timing, delay=0, site=None, ephem=None, interp=False, step=10):
	if site is None: site = coordinates.default_site
	tdb  = tdt2tdb(tai2tdt(utc2tai(ctime)))
	tdb -= calc_obs_delay(ctime, target, site, ephem=ephem, interp=interp, step=step)
	tdb -= delay
	ind, x = pulsar_timing.calc_ind_x(tdb)
	phase  = x % 1
	return phase

class PulsarTiming:
	def __init__(self, fname):
		# Consider reading in the dispersion delay too. It's totally irrelevant in the mm though
		data = np.loadtxt(fname, usecols=(3,4,6,8), ndmin=2).T
		self.tref  = utils.mjd2ctime(data[0])
		self.t0    = data[1]
		self.freq  = data[2]
		self.dfreq = data[3]*1e-15
		self.P     = 1/self.freq
		self.dP    = -1/self.freq**2 * self.dfreq
	def calc_ind_x(self, ctime):
		ind = np.searchsorted(self.tref, ctime, side="right")-1
		return ind, calc_x(ctime-(self.tref[ind]+self.t0[ind]), self.P[ind], self.dP[ind])
	def calc_t(self, ind, x):
		return calc_t(x, self.P[ind], self.dP[ind])+(self.tref[ind]+self.t0[ind])

def calc_obs_delay(ctime, target, site, ephem=None, interp=False, step=10):
	from astropy.time import Time
	from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
	if interp:
		t1, t2 = utils.minmax(ctime)
		tsamp  = np.linspace(t1, t2, utils.ceil((t2-t1)/step))
		dsamp  = calc_obs_delay(tsamp, target, site, interp=False)
		return utils.interp(ctime, tsamp, dsamp)
	# Vectorization
	ctime = np.asarray(ctime)
	shape = ctime.shape
	ctime = ctime.reshape(-1)
	# Must use this ephemeris to match crabtime assumptions:
	# https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de200.bsp
	if ephem: solar_system_ephemeris.set(ephem)
	# The direction towards the pulsar. Independent of position in
	# solar system due to huge distance.
	vec_obsdir    = utils.ang2rect(target)
	# Get vector from earth center to observer
	obs_ra, obs_dec = coordinates.transform("hor", "cel", [0,np.pi/2], time=utils.ctime2mjd(ctime), site=site)
	vec_earth_obs = utils.ang2rect([obs_ra, obs_dec])*utils.R_earth
	# Vector from solar system barycenter to Earth
	vec_bary_earth= get_body_barycentric("earth", Time(ctime, format="unix")).xyz.to("m").value
	# Want component of obs-bary vector along vec_obsdir
	vec_bary_obs  = vec_bary_earth + vec_earth_obs
	vec_obs_bary  = -vec_bary_obs
	delay         = np.sum(vec_obs_bary*vec_obsdir[:,None],0) / utils.c
	return delay.reshape(shape)

# Helper functions

def load_pulsar_timing(fname):
	data = np.loadtxt(fname, usecols=(3,4,6,8), ndmin=2).T
	return bunch.Bunch(tref=utils.mjd2ctime(data[0]), t0=data[1], freq=data[2], dfreq=data[3]*1e-15)
# Phase/time mapping
def calc_t(x, P, dP): return P/dP*np.expm1(dP*x)
def calc_x(t, P, dP): return np.log1p(dP/P*t)/dP
def predict_t0(t_targ, t, nu, dnu):
	P    = 1/nu
	dP   = -1/nu**2*dnu
	tarriv = calc_t(np.ceil(calc_x(t_targ-t,P,dP)),P,dP)+t
	# Relative to midnight
	return tarriv % utils.day
# Time remapping
class UTCTAI:
	def __init__(self):
		iers_mjds = np.array([iers.get(i).mjd for i in range(iers.cvar.iers_n)])
		dUTs      = np.array([iers.get(i).dUT for i in range(iers.cvar.iers_n)])
		leaps     = 12+np.concatenate([[0],np.cumsum(utils.nint(dUTs[1:]-dUTs[:-1]))])
		leap_inds = np.where(utils.nint(dUTs[1:]-dUTs[:-1])!=0)[0]+1
		self.leap_mjds = iers_mjds[leap_inds]
		self.leap_vals = 12+np.cumsum(utils.nint(dUTs[1:]-dUTs[:-1]))[leap_inds-1]
_utctai = None
# Many of these functions are not exact inverses, but they are very close to it
def calc_leaps(ctime):
	global _utctai
	if _utctai is None: _utctai = UTCTAI()
	ind = np.maximum(np.searchsorted(_utctai.leap_mjds, utils.ctime2mjd(ctime), side="right")-1,0)
	return _utctai.leap_vals[ind]
def tai2utc(ctime): return ctime - calc_leaps(ctime)
def utc2tai(ctime): return ctime + calc_leaps(ctime)
def tai2tdt(ctime): return ctime + 32.184
def tdt2tai(ctime): return ctime - 32.184
def calc_tdb_off(ctime):
	jd = utils.ctime2jd(ctime)
	T  = (jd-2451545.0)/365.25
	g  = 2*np.pi*(357.528+359.99050*T)/360
	return 0.001658*np.sin(g+0.0167*np.sin(g))
def tdt2tdb(ctime): return ctime + calc_tdb_off(ctime)
def tdb2tdt(ctime): return ctime - calc_tdb_off(ctime)
