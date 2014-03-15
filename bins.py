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

def expbin(n, nbin=None, nmin=8):
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
	tmp = np.array(fixed)
	tmp[-1] = n
	return np.vstack((tmp[:-1],tmp[1:])).T
