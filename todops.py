import numpy as np
from enlib import utils, pmat

def fit_phase_flat(tods, az, daz=1*utils.arcmin, cuts=None, niter=3, overlap=10, clean_tod=False):
	# for the given tods[ndet,nsamp], cuts (multirange[ndet,nsamp]) and az[nsamp],
	if not clean_tod: tods = tods.copy()
	# Set up phase pixels
	amin = np.min(az)
	amax = np.max(az)
	naz = int((amax-amin)/daz)+1
	pflat = pmat.PmatPhaseFlat(amin, daz, naz)
	# Output and work arrays
	phase  = np.zeros((2,naz))
	dphase = phase.copy()
	hits   = phase.copy()
	# Precompute div
	pflat.backard(tods*0+1, hits, -1)
	hits[hits==0] = 1
	for i in range(niter):
		# Overall logic: gapfill -> bin -> subtract -> loop
		if cuts is not None:
			gapfill.gapfill_linear(tods, cuts, overlap=overlap, inplace=True)
		pflat.backward(tods, dphase)
		dphase /= hits
		phase += dphase
		pflat.forward(tods, -dphase)
	return phase
