"""Point source parameter I/O. In order to simulate a point source as it appears on
the sky, we need to know its position, amplitude and local beam shape (which can
also absorb an extendes size for the source, as long as it's gaussian). While other
properties may be nice to know, those are the only ones that matter for simulating
it. This module provides functions for reading these minimal parameters from
various data files.

The standard parameters are:
	dec (radians)
	ra (radians)
	[T,Q,U] amplitude at center of gaussian (uK)
	beam sigma (wide  axis) (radians)
	beam sigma (short axis) (radians)
	beam orientation (wide axis from dec axis)  (radians)
"""
import numpy as np
from enlib import utils

def read_rahul_marius(fname, exact=False, default_beam=1*utils.arcmin, flux_factor=500):
	"""This format has no beam information, and lists only fluxes.
	Beams will be set to a default 1', and the flux will be converted
	to amplitude with a very rough approximation."""
	vals = []
	with open(fname, "r") as f:
		for line in f:
			if len(line) == 0 or line[0] == '#': continue
			toks = line.split()
			ra, dec = [float(toks[i]) for i in [2,3]]
			ra_true, dec_true = [parse_angle_sexa(toks[i]) for i in [11,12]]
			ra_true *= 15
			flux = np.array([float(toks[8]), 0, 0])
			has_exact = int(toks[10]) >= 0
			if exact:
				if not has_exact: continue
				print ra_true, dec_true, ra,dec
				ra, dec = ra_true, dec_true
			vals.append([dec*utils.degree, ra*utils.degree]+list(flux*flux_factor)+[default_beam,default_beam,0])
	return np.array(vals)

def read_skn(fname, default_beam=1*utils.arcmin):
	tmp = np.loadtxt(fname)
	print tmp.shape
	if tmp.shape[1] == 5: return read_skn_posamp(fname, default_beam)
	elif tmp.shape[1] == 8: return read_skn_standard(fname)
	elif tmp.shape[1] == 25: return read_skn_full(fname)
	else: raise IOError("Unrecognized skn format")

def read_skn_posamp(fname, default_beam=1*utils.arcmin):
	"""dec ra T Q U"""
	tmp = np.loadtxt(fname)
	b   = np.full(len(tmp),default_beam)
	return np.concatenate([tmp[:,:2]*utils.degree,tmp[:,3:6],b[:,None],b[:,None],b[:,None]*0],1)

def read_skn_standard(fname):
	"""dec ra T Q U bw bh phi"""
	tmp = np.loadtxt(fname)
	return np.concatenate([tmp[:,:2]*utils.degree,tmp[:,2:5],
		tmp[:,5:7]*utils.arcmin*utils.fwhm, tmp[:,7:8]*utils.degree],1)

def read_skn_full(fname):
	"""id rank S/N dec ddec ra dra T dT Q dQ U dU Tf dTf Qf dQf Uf dUf bw dbw bh dbh phi dphi"""
	tmp = np.loadtxt(fname)
	return np.concatenate([tmp[:,3:7:2]*utils.degree, tmp[:,7:13:2],
		tmp[:,19:23:2]*utils.arcmin*utils.fwhm, tmp[:,23:25:2]*utils.degree],1)

def parse_angle_sexa(s):
	sign = 1
	if s[0] == "-":
		s, sign = s[1:], -1
	return sign*np.sum([float(w)*60.0**(-i) for i,w in enumerate(s.split(":"))])
