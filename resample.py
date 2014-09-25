"""This module handles resampling of time-series and similar arrays."""
import numpy as np
from enlib import utils, fft

def resample(d, factors=[0.5], axes=None, method="fft"):
	if np.allclose(factors,1): return d
	if method == "fft":
		return resample_fft(d, factors, axes)
	elif method == "bin":
		return resample_bin(d, factors, axes)
	else:
		raise NotImplementedError("Resampling method '%s' is not implemented" % method)

def resample_bin(d, factors=[0.5], axes=None):
	if np.allclose(factors,1): return d
	down = [max(1,int(round(1/f))) for f in factors]
	up   = [max(1,int(round(f)))   for f in factors]
	d    = downsample_bin(d, down, axes)
	return upsample_bin  (d, up, axes)

def downsample_bin(d, steps=[2], axes=None):
	assert len(steps) <= d.ndim
	if axes is None: axes = np.arange(-1,-len(steps)-1,-1)
	assert len(axes) == len(steps)
	# Expand steps to cover every axis in order
	fullsteps = np.zeros(d.ndim,dtype=int)+1
	for ax, step in zip(axes, steps): fullsteps[ax]=step
	# Make each axis an even number of steps to prepare for reshape
	s = tuple([slice(0,L/step*step) for L,step in zip(d.shape,fullsteps)])
	d = d[s]
	# Reshape each axis to L/step,step to prepare for mean
	newshape = np.concatenate([[L/step,step] for L,step in zip(d.shape,fullsteps)])
	d = np.reshape(d, newshape)
	# And finally take the mean over all the extra axes
	return np.mean(d, tuple(range(1,d.ndim,2)))

def upsample_bin(d, steps=[2], axes=None):
	shape = d.shape
	assert len(steps) <= d.ndim
	if axes is None: axes = np.arange(-1,-len(steps)-1,-1)
	assert len(axes) == len(steps)
	# Expand steps to cover every axis in order
	fullsteps = np.zeros(d.ndim,dtype=int)+1
	for ax, step in zip(axes, steps): fullsteps[ax]=step
	# Reshape each axis to (n,1) to prepare for tiling
	newshape = np.concatenate([[L,1] for L in shape])
	d = np.reshape(d, newshape)
	# And tile
	d = np.tile(d, np.concatenate([[1,s] for s in fullsteps]))
	# Finally reshape back to proper dimensionality
	return np.reshape(d, np.array(shape)*np.array(fullsteps))

def resample_fft(d, factors=[0.5], axes=None):
	"""Resample numpy array d via fourier-reshaping. Requires periodic data.
	"factors" indicates the factors by which the axis lengths should be
	increased. If less factors are specified than the number of axes,
	the numbers apply to the last N axes, unless the "axes" argument
	is used to specify which ones."""
	if np.allclose(factors,1): return d
	factors = np.atleast_1d(factors)
	assert len(factors) <= d.ndim
	if axes is None: axes = np.arange(-len(factors),0)
	assert len(axes) == len(factors)
	fd = fft.fft(d, axes=axes)
	# Frequencies are 0 1 2 ... N/2 (-N)/2 (-N)/2+1 .. -1
	# Ex 0* 1 2* -1 for n=4 and 0* 1 2 -2 -1 for n=5
	# To upgrade,   insert (n_new-n_old) zeros after n_old/2
	# To downgrade, remove (n_old-n_new) values after n_new/2
	# The idea is simple, but arbitrary dimensionality makes it
	# complicated.
	for ax, factor in zip(axes, factors):
		ax %= d.ndim
		nold = d.shape[ax]
		nnew = int(nold*factor)
		dn   = nnew-nold
		if dn > 0:
			padvals = np.zeros(fd.shape[:ax]+(dn,)+fd.shape[ax+1:])
			spre  = tuple([slice(None)]*ax+[slice(0,nold/2)]+[slice(None)]*(fd.ndim-ax-1))
			spost = tuple([slice(None)]*ax+[slice(nold/2,None)]+[slice(None)]*(fd.ndim-ax-1))
			fd = np.concatenate([fd[spre],padvals,fd[spost]],axis=ax)
		elif dn < 0:
			spre  = tuple([slice(None)]*ax+[slice(0,nnew/2)]+[slice(None)]*(fd.ndim-ax-1))
			spost = tuple([slice(None)]*ax+[slice(nnew/2-dn,None)]+[slice(None)]*(fd.ndim-ax-1))
			fd = np.concatenate([fd[spre],fd[spost]],axis=ax)
	# And transform back
	res  = fft.ifft(fd, axes=axes, normalize=True)
	del fd
	res *= np.product(factors)
	return res if np.issubdtype(d.dtype, np.complexfloating) else res.real
