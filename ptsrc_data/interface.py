import numpy as np, fortran_32, fortran_64

def get_core(dtype):
	if dtype == np.float32:
		return fortran_32.fortran
	elif dtype == np.float64:
		return fortran_64.fortran
	raise NotImplementedError

def nmat_mwhite(tod, noise, submean=2):
	"""Applies white noise + mean subtraction noise model to tod, overwriting it."""
	core = get_core(tod.dtype)
	core.nmat_mwhite(tod, noise.ranges.T, noise.rangesets, noise.offsets.T, noise.ivars, submean)
	return tod

def measure_mwhite(tod, data, submean=2):
	core = get_core(tod.dtype)
	nsrc, ndet = data.offsets[:,:-1].shape
	vars  = np.zeros([nsrc,ndet],dtype=tod.dtype)
	nvars = np.zeros([nsrc,ndet],dtype=np.int32)
	core.measure_mwhite(tod, data.ranges.T, data.rangesets, data.offsets.T, vars.T, nvars.T, submean)
	return vars, nvars

def pmat_thumbs(dir, tod, maps, point, phase, boxes):
	core = get_core(tod.dtype)
	core.pmat_thumbs(dir, tod.T, maps.T, point.T, phase.T, boxes.T)
