"""This module provides functions for filling gaps in an array based on ranges or masks."""
import numpy as np
from enlib.rangelist import Rangelist

def gapfill_linear(arr, ranges):
	"""Returns arr with the ranges given by ranges, which can be [:,{from,to}] or
	a Rangelist, filled using linear interpolation."""
	ranges = Rangelist(ranges, len(arr), copy=False)
	res = arr.copy()
	for r1,r2 in ranges.ranges:
		if r1 == 0 and r2 == len(ranges):
			res[r1:r2] = 0
		elif r1 == 0:
			res[r1:r2] = res[r2]
		elif r2 == len(ranges):
			res[r1:r2] = res[r1-1]
		else:
			res[r1-1:r2] = np.linspace(res[r1],res[r2],r2-r1,endoint=False)
	return res

def gapfill_copy(arr, ranges, overlap=10):
	"""Returns arr with the ranges given by ranges, which can be [:,{from,to}] or
	a Rangelist, filled using resloped copies of parts of the uncut data. This
	results in less bias in the noise properties of the resulting stream than
	smooth interpolation, though it is not as good as filling with a constrained
	realization. The overlap parameter specifies the number of samples to use
	on each side when computing the slope and offset. Smaller values produce a noisier
	estimate of these values, larger values degrade resolution."""
	ranges = Rangelist(ranges, len(arr), copy=False)
	uncut  = ranges.invert()
	longest_uncut = np.argmax(uncut.ranges[:,1]-uncut.ranges[:,0])
	u1,u2  = uncut.ranges[longest_uncut]
	for r1,r2 in ranges.ranges:
		filler = repeat_filler(arr[u1:u2], r2-r1+2*overlap)
		# Reslope so that we match the overlap regions. For edge cases,
		# use the other side.
		left  = np.mean(arr[max(0,r1-overlap):r1]) if r1 > 0 else None
		right = np.mean(arr[r2:min(len(arr),r2+overlap)]) if r2 < len(tod) else left
		if left == None:
			left = right
		uleft = np.mean(filler[:overlap])
		uright= np.mean(filler[-overlap:])
		arr[r1:r2] = filler[overlap:-overlap] + ((left-uleft) + np.arange(r2-r1)*((right-uright)-(left-uleft))/(r2-r1))
	return arr
