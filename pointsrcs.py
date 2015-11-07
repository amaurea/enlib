"""Point source parameter I/O. In order to simulate a point source as it appears on
the sky, we need to know its position, amplitude and local beam shape (which can
also absorb an extendes size for the source, as long as it's gaussian). While other
properties may be nice to know, those are the only ones that matter for simulating
it. This module provides functions for reading these minimal parameters from
various data files.

The standard parameters are [nsrc,nparam]:
	dec (radians)
	ra (radians)
	[T,Q,U] amplitude at center of gaussian (uK)
	beam sigma (wide  axis) (radians)
	beam sigma (short axis) (radians)
	beam orientation (wide axis from dec axis)  (radians)
"""
import numpy as np
from enlib import utils

def read(fname, format="auto", exact=None, default_beam=1*utils.arcmin, amp_factor=1/395.11):
	"""Try to read a point source file in any format."""
	if format == "skn": return read_skn(fname, default_beam=default_beam)
	elif format == "rahul_marius": return read_rahul_marius(fname, exact=exact, default_beam=default_beam, amp_factor=amp_factor)
	elif format == "auto":
		try:
			return read_skn(fname, default_beam=default_beam)
		except (IOError, ValueError):
			return read_rahul_marius(fname, exact=exact, default_beam=default_beam, amp_factor=amp_factor)
	else:
		raise ValueError("Unrecognized format '%s'" % format)

def write(fname, srcs, format="auto"):
	if format == "skn" or format == "auto":
		write_skn_standard(fname, srcs)
	else: ValueError("Unrecognized format '%s'" % format)

def read_rahul_marius(fname, exact=None, default_beam=1*utils.arcmin, amp_factor=1/395.11):
	"""This format has no beam information, and lists only T amps in Jy/steradian.
	Beams will be set to a default 1', and the amps will be converted
	to amplitude. Default is to read only the
	data for which confirmed exact positions are available, as that is the
	purpose of the rahul marius lists."""
	vals = []
	if exact is None: exact = True
	with open(fname, "r") as f:
		for line in f:
			if len(line) == 0 or line[0] == '#': continue
			toks = line.split()
			if len(toks) != 13 and len(toks) != 15: raise IOError("File is not in rahul marius format")
			ra, dec = [float(toks[i]) for i in [2,3]]
			amp = np.array([float(toks[4]), 0, 0])
			if exact:
				has_exact = int(toks[10]) >= 0
				if not has_exact: continue
				ra_true, dec_true = float(toks[13]), float(toks[14])
				ra, dec = ra_true, dec_true
			vals.append([dec*utils.degree, ra*utils.degree]+list(amp*amp_factor)+[default_beam,default_beam,0])
	return np.array(vals)

def read_skn(fname, default_beam=1*utils.arcmin):
	tmp = np.loadtxt(fname)
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

def write_skn_standard(fname, srcs):
	deg = utils.degree
	fwhm = utils.arcmin*utils.fwhm
	with open(fname, "w") as f:
		for src in srcs:
			f.write("%10.5f %10.5f %10.3f %10.3f %10.3f %7.4f %7.4f %7.2f\n" % (
				src[0]/deg, src[1]/deg, src[2], src[3], src[4],
				src[5]/fwhm, src[6]/fwhm, src[7]/deg))

def parse_angle_sexa(s):
	"""Parse an angle in the form [+-]deg:min:sec"""
	sign = 1
	if s[0] == "-":
		s, sign = s[1:], -1
	return sign*np.sum([float(w)*60.0**(-i) for i,w in enumerate(s.split(":"))])

def src2param(srcs):
	"""For fast source model evaluation, it is useful to store the beam as an inverse
	covariance matrix."""
	params = np.array(srcs)
	if params.ndim == 1: return src2param(params[None])[0]
	params[:,5:8] = np.array([utils.compress_beam(b[:2],b[2]) for b in srcs[:,5:8]])
	return params

def param2src(params):
	srcs = np.array(params)
	if srcs.ndim == 1: return param2src(srcs[None])[0]
	for src in srcs:
		sigma, phi = utils.expand_beam(src[5:8])
		src[5:7] = sigma
		src[7] = phi
	return srcs
