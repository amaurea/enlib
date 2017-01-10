"""This module defines functions for various binning schemes."""
import numpy as np

def linbin(n, nbin=None, nmin=None):
	"""Given a number of points to bin and the number of approximately
	equal-sized bins to generate, returns [nbin_out,{from,to}].
	nbin_out may be smaller than nbin. The nmin argument specifies
	the minimum number of points per bin, but it is not implemented yet.
	nbin defaults to the square root of n if not specified."""
	if not nbin: nbin = int(np.round(n**0.5))
	tmp  = np.arange(nbin+1)*n/nbin
	return np.vstack((tmp[:-1],tmp[1:])).T

def expbin(n, nbin=None, nmin=8, nmax=0):
	"""Given a number of points to bin and the number of exponentially spaced
	bins to generate, returns [nbin_out,{from,to}].
	nbin_out may be smaller than nbin. The nmin argument specifies
	the minimum number of points per bin. nbin defaults to n**0.5"""
	if not nbin: nbin = int(np.round(n**0.5))
	tmp  = np.array(np.exp(np.arange(nbin+1)*np.log(n+1)/nbin)-1,dtype=int)
	fixed = [tmp[0]]
	i = 0
	while i < nbin:
		for j in range(i+1,nbin+1):
			if tmp[j]-tmp[i] >= nmin:
				fixed.append(tmp[j])
				i = j
	# Optionally split too large bins
	if nmax:
		tmp = [fixed[0]]
		for v in fixed[1:]:
			dv = v-tmp[-1]
			nsplit = (dv+nmax-1)/nmax
			tmp += [tmp[-1]+dv*(i+1)/nsplit for i in range(nsplit)]
		fixed = tmp
	tmp = np.array(fixed)
	tmp[-1] = n
	return np.vstack((tmp[:-1],tmp[1:])).T

def bin_data(bins, d, op=np.mean):
	"""Bin the data d into the specified bins along the last dimension. The result has
	shape d.shape + (nbin,)."""
	nbin  = bins.shape[0]
	dflat = d.reshape(-1,d.shape[-1])
	dbin  = np.zeros([dflat.shape[0], nbin])
	for bi, b in enumerate(bins):
		dbin[:,bi] = op(dflat[:,b[0]:b[1]],1)
	return dbin.reshape(d.shape[:-1]+(nbin,))

def bin_expand(bins, bdata):
	res = np.zeros(bdata.shape[:-1]+(bins[-1,1],),bdata.dtype)
	for bi, b in enumerate(bins):
		res[...,b[0]:b[1]] = bdata[...,bi]
	return res
