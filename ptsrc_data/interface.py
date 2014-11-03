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
	rangemask = np.zeros(noise.ranges.shape[0],dtype=np.int32)+1
	core.nmat_mwhite(tod, noise.ranges.T, noise.rangesets, noise.offsets.T, noise.ivars, submean, rangemask)
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

def pmat_model(tod, params, data, dir=1):
	core = get_core(tod.dtype)
	rangemask = np.zeros(data.ranges.shape[0],dtype=np.int32)+1
	core.pmat_model(dir, tod, params.T, data.ranges.T, data.rangesets.T, data.offsets.T, data.point.T, data.phase.T, rangemask)

def chisq_by_range(tod, params, data, submean=2, prev_params=None, prev_chisqs=None):
	changed = np.zeros(params.shape,dtype=bool)+True if prev_params is None else params != prev_params
	if not np.any(changed): return prev_chisqs
	# Check which sources have changed
	core = get_core(tod.dtype)
	changed_srcs   = np.any(changed,axis=1).astype(np.int32)
	changed_ranges = np.zeros(data.ranges.shape[0],np.int32)
	core.srcmask2rangemask(changed_srcs, data.rangesets.T, data.offsets.T, changed_ranges)
	# Compute the chisquare for the changed ranges
	wtod = np.empty(tod.shape, tod.dtype)
	core.pmat_model(1, wtod, params.T, data.ranges.T, data.rangesets.T, data.offsets.T, data.point.T, data.phase.T, changed_ranges)
	core.rangesub(wtod, tod, data.ranges.T, changed_ranges)
	ntod = wtod.copy()
	core.nmat_mwhite(ntod, data.ranges.T, data.rangesets.T, data.offsets.T, data.ivars, submean, changed_ranges)
	chisqs = np.zeros(data.ranges.shape[0],dtype=np.float64)
	core.rangechisq(wtod, ntod, data.ranges.T, chisqs, changed_ranges)
	# Fill in old chisquares for those that didn't change
	if prev_params is not None:
		chisqs[changed_ranges==0] = prev_chisqs[changed_ranges==0]
	return chisqs

def chisq(tod, params, data, submean=2):
	core = get_core(tod.dtype)
	wtod = np.empty(tod.shape, tod.dtype)
	pmat_model(wtod, params, data)
	wtod -= tod
	ntod = wtod.copy()
	nmat_mwhite(ntod, data, submean)
	return np.sum(wtod*ntod)
