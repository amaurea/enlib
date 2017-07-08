"""This module provides functions for filling gaps in an array based on ranges or masks."""
import numpy as np, utils
from enlib import fft, config, resample
from enlib.utils import repeat_filler
from enlib.rangelist import Rangelist, Multirange, multify

config.default("gapfill", "linear", "TOD gapfill method. Can be 'copy' or 'linear'")
config.default("gapfill_context", 10, "Samples of context to use for matching up edges of cuts.")

def gapfill(arr, ranges, inplace=False, overlap=None):
	gapfiller = {
			"linear":gapfill_linear,
			"joneig":gapfill_joneig,
			"copy":gapfill_copy,
			"cubic":gapfill_cubic
		}[config.get("gapfill")]
	overlap = config.get("gapfill_context", overlap)
	return gapfiller(arr, ranges, inplace=inplace, overlap=overlap)

@multify
def gapfill_linear(arr, ranges, inplace=False, overlap=None):
	"""Returns arr with the ranges given by ranges, which can be [:,{from,to}] or
	a Rangelist, filled using linear interpolation."""
	ranges = Rangelist(ranges, len(arr), copy=False)
	overlap= config.get("gapfill_context", overlap)
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
			arr[r1:r2] = np.linspace(np.mean(arr[left:r1]), np.mean(arr[r2:right]), r2-r1+1,endpoint=False)[1:]
	return arr

def gapfill_pair(tod, cut, inplace=False, gapfill=gapfill_linear):
	"""Gapfill a set of tod-pairs tod[npair*2,nsamp] in the cut regions
	cut[npair*2,{ranges}], by gapfilling pair-difference rather than the
	individual timestreams. Where only one det in a pair is cut, the
	gapfilled pairdiff is used to recover the other. Where both are cut,
	gapfilling is used on one of the detectors to give a baseline signal."""
	if not inplace: tod = tod.copy()
	# Split tods and cuts into the two members of our pair
	d0,d1 = tod[0::2], tod[1::2]
	c0,c1 = cut[0::2], cut[1::2]
	# cut union for diff gapfilling, cut intersection for non-diff gapfilling
	# and cut difference for diff-based reconstruction.
	cut_union = c0+c1
	cut_inter = (c0.invert()+c0.invert()).invert()
	cut0n1    = (c0.invert()+c0).invert()
	# First gapfill the pair difference
	dtod = d0-d1
	dtod = gapfill(dtod,cut_union)
	# Then fill build the cut parts of one tod based on the corresponding
	# uncut parts of the other, and the gapfilled diff
	cut0n1.insert(d0,cut0n1.extract(d1)+cut0n1.extract(dtod))
	# Gapfill the samples the previous method couldn't get us
	d0 = gapfill(d0,cut_inter)
	# And reconstruct the other tod the same way
	cut_union.insert(d1,cut_union.extract(d0)-cut_union.extract(dtod))
	# Copy back over into a single array
	tod[0::2] = d0
	tod[1::2] = d1
	return tod

def fit_linear(arr, ref=0, nsigma=2):
	"""Fit b+ax to the array centered on a point a fraction ref
	along the array. Bias the slope towards zero by nsigma standard
	deviations."""
	if len(arr) == 0: return np.array([0,0])
	elif len(arr) == 1: return np.array([arr[0],0])
	else:
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

def gapfill_joneig(tod, cut, thresh=4, niter=4, nloop=4, inplace=False, gapfill=gapfill_linear, cov_step=10, amp_step=10, overlap=None):
	"""Gapfill a tod[ndet,nsamp] in cuts cut[ndet,{ranges}] using
	Jon's eigenmode iteration. It's about 20 times slower than linear
	gapfilling, mostly due to calling gapfill_linear niter*nloop times
	internally."""
	tod = gapfill(tod, cut, inplace=inplace)
	cut_small = cut[:,::amp_step]
	for i in range(nloop):
		# Find the most important basis vectors for the current data
		sub = np.ascontiguousarray(tod[:,::cov_step])
		cov = sub.dot(sub.T)
		cov = 0.5*(cov+cov.T)
		e, v = np.linalg.eigh(cov)
		# v is [ndet,nmode]
		mask = e > (thresh**2*np.median(e))
		e, v = e[mask], v[:,mask]
		basis = v.T.dot(tod) # [nmode, nsamp]
		# Iteratively subtract best basis vector fit and 
		work_small  = resample.downsample_bin(tod, [amp_step], [-1])
		basis_small = resample.downsample_bin(basis, [amp_step], [-1])
		div = basis_small.dot(basis_small.T)
		amps_tot = 0
		for j in range(niter):
			amps = np.linalg.solve(div, basis_small.dot(work_small.T))
			amps_tot += amps
			work_small -= amps.T.dot(basis_small)
			gapfill(work_small, cut_small, inplace=True)
		tod -= amps_tot.T.dot(basis)
		gapfill(tod, cut, inplace=True)
		tod += amps_tot.T.dot(basis)
	return tod

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
