"""This module provides functions for filling gaps in an array based on ranges or masks."""
import numpy as np, utils
from enlib import fft
from enlib.utils import repeat_filler
from enlib.rangelist import Rangelist, Multirange, multify

@multify
def gapfill_linear(arr, ranges, inplace=False, overlap=1):
	"""Returns arr with the ranges given by ranges, which can be [:,{from,to}] or
	a Rangelist, filled using linear interpolation."""
	ranges = Rangelist(ranges, len(arr), copy=False)
	if not inplace: arr = np.array(arr)
	nr = len(ranges.ranges)
	n  = ranges.n
	for i, (r1,r2) in enumerate(ranges.ranges):
		left  = max(0 if i == 0    else ranges.ranges[i-1,1],r1-overlap)
		right = min(n if i == nr-1 else ranges.ranges[i+1,0],r2+overlap)
		# If the cut coveres the whole array, fill with 0
		if r1 == 0 and r2 == len(arr):
			arr[r1:r2] = 0
		# If it goes all the way to one end, use the value from one side
		elif r1 == 0:
			arr[r1:r2] = np.mean(arr[r2:right])
		elif r2 == len(arr):
			arr[r1:r2] = np.mean(arr[left:r1])
		# Otherwise use linear interpolation
		else:
			arr[r1-1:r2] = np.linspace(np.mean(arr[left:r1]), np.mean(arr[r2:right]), r2-r1+1,endpoint=False)
	return arr

def fit_linear(arr, ref=0, nsigma=2):
	"""Fit b+ax to the array centered on a point a fraction ref
	along the array. Bias the slope towards zero by nsigma standard
	deviations."""
	N = np.mean((arr[1:]-arr[:-1])**2)/2
	B = np.full((2,len(arr)),1.0)
	B[1] = np.arange(len(arr))-0.5*len(arr)
	res = np.linalg.solve(B.dot(B.T),B.dot(arr))
	# Estimate uncertainty
	cov = np.linalg.inv(B.dot(B.T))*N
	std = np.diag(cov)**0.5
	# Reduce slope amplitude by up to 1 sigma, to avoid huge
	# fluctuations in noisy regions
	if res[1] > 0: res[1] = np.maximum(res[1]-nsigma*std[1],0)
	else:          res[1] = np.minimum(res[1]+nsigma*std[1],0)
	# Use the fixed slope to move us to the reference point
	res[0] += res[1]*(ref-0.5)
	return res
def generate_cubic(p1, p2, n):
	coeff = np.linalg.solve([
		[   1,   0,   0,   0   ],
		[   1,   n,   n**2,n**3],
		[   0,   1,   0,   0   ],
		[   0,   1, 2*n, 3*n**2]],
		[p1[0],p2[0],p1[1],p2[1]])
	inds = np.arange(n)
	return coeff[0] + coeff[1]*inds + coeff[2]*inds**2 + coeff[3]*inds**3

@multify
def gapfill_cubic(arr, ranges, inplace=False, overlap=10):
	"""Returns arr with the ranges given by ranges, which can be [:,{from,to}] or
	a Rangelist, filled using cubic interpolation."""
	ranges = Rangelist(ranges, len(arr), copy=False)
	if not inplace: arr = np.array(arr)
	nr = len(ranges.ranges)
	n  = ranges.n
	for i, (r1,r2) in enumerate(ranges.ranges):
		left  = max(0 if i == 0    else ranges.ranges[i-1,1],r1-overlap)
		right = min(n if i == nr-1 else ranges.ranges[i+1,0],r2+overlap)
		# If the cut coveres the whole array, fill with 0
		if r1 == 0 and r2 == len(arr):
			arr[r1:r2] = 0
		# If it goes all the way to one end, use the value from one side
		elif r1 == 0:
			p = fit_linear(arr[r2:right],0)
			arr[r1:r2] = generate_cubic([p[0],0],p,r2-r1)
		elif r2 == len(arr):
			p = fit_linear(arr[left:r1],1)
			arr[r1:r2] = generate_cubic(p, [p[0],0], r2-r1)
		else:
			p1 = fit_linear(arr[left:r1 ],1)
			p2 = fit_linear(arr[r2:right],0)
			arr[r1:r2] = generate_cubic(p1,p2,r2-r1)
	return arr

@multify
def gapfill_constant(arr, ranges, inplace=False, value=0.0, overlap=None):
	"""Returns arr with the ranges given by ranges, which can be [:,{from,to}] or
	a Rangelist, filled using a constant value."""
	ranges = Rangelist(ranges, len(arr), copy=False)
	if not inplace: arr = np.array(arr)
	for r1,r2 in ranges.ranges:
		arr[r1:r2] = value
	return arr

@multify
def gapfill_copy(arr, ranges, overlap=10, inplace=False):
	"""Returns arr with the ranges given by ranges, which can be [:,{from,to}] or
	a Rangelist, filled using resloped copies of parts of the uncut data. This
	results in less bias in the noise properties of the resulting stream than
	smooth interpolation, though it is not as good as filling with a constrained
	realization. The overlap parameter specifies the number of samples to use
	on each side when computing the slope and offset. Smaller values produce a noisier
	estimate of these values, larger values degrade resolution."""
	if not inplace: arr = np.array(arr)
	ranges = Rangelist(ranges, len(arr), copy=False)
	uncut  = ranges.invert()
	longest_uncut = np.argmax(uncut.ranges[:,1]-uncut.ranges[:,0])
	u1,u2  = uncut.ranges[longest_uncut]
	for r1,r2 in ranges.ranges:
		filler = repeat_filler(arr[u1:u2], r2-r1+2*overlap)
		# Reslope so that we match the overlap regions. For edge cases,
		# use the other side. Do a cut-weighted average to avoid inluding
		# data from other cuts in these means.
		left  = mean_cut_range(arr, ranges, [max(0,r1-overlap),r1]) if r1 > 0 else None
		right = mean_cut_range(arr, ranges, [r1,min(len(arr),r2+overlap)]) if r2 < len(arr) else left
		if left == None:
			left = right
		uleft = np.mean(filler[:overlap])
		uright= np.mean(filler[-overlap:])
		arr[r1:r2] = filler[overlap:-overlap] + ((left-uleft) + np.arange(r2-r1)*((right-uright)-(left-uleft))/(r2-r1))
	return arr

def gapfill_values(arr, ranges, values, inplace=False):
	"""Return arr with the gaps filled with values copied from the corresponding
	locations in the given array."""
	if not inplace: arr = np.array(arr)
	ndet = arr.shape[0]
	for d in range(ndet):
		for i, (r1,r2) in enumerate(ranges[d].ranges):
			arr[d,r1:r2] = values[d,r1:r2]
	return arr

def mean_cut_range(a, c, r=[None,None]):
	a = a[r[0]:r[1]]
	c = c[r[0]:r[1]]
	mask = 1-c.to_mask()
	n = np.sum(mask)
	if n > 0: return np.sum(a*mask)/n
