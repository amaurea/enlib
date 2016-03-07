"""This module provides functions for filling gaps in an array based on ranges or masks."""
import numpy as np
from enlib.utils import repeat_filler
from enlib.rangelist import Rangelist, Multirange, multify

@multify
def gapfill_linear(arr, ranges, inplace=False):
	"""Returns arr with the ranges given by ranges, which can be [:,{from,to}] or
	a Rangelist, filled using linear interpolation."""
	ranges = Rangelist(ranges, len(arr), copy=False)
	if not inplace: arr = np.array(arr)
	for r1,r2 in ranges.ranges:
		if r1 == 0 and r2 == len(ranges):
			arr[r1:r2] = 0
		elif r1 == 0:
			arr[r1:r2] = arr[r2]
		elif r2 == len(ranges):
			arr[r1:r2] = arr[r1-1]
		else:
			arr[r1-1:r2] = np.linspace(arr[r1-1],arr[r2],r2-r1+1,endpoint=False)
	return arr

@multify
def gapfill_constant(arr, ranges, inplace=False, value=0.0):
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

def mean_cut_range(a, c, r=[None,None]):
	a = a[r[0]:r[1]]
	c = c[r[0]:r[1]]
	mask = 1-c.to_mask()
	n = np.sum(mask)
	if n > 0: return np.sum(a*mask)/n
