"""This module handles deprojection of a set of arrays from another set of
arrays. This is useful for cleaning TODs of unwanted signals, for example."""
import numpy as np, scipy.signal
from enlib import utils, pmat, sampcut, gapfill, fft

def estimate_white_noise(tod, nchunk=10, chunk_size=1000):
	"""Robust time-domain estimation of white noise level."""
	vs = []
	for ci in range(nchunk):
		i1 = ci*tod.shape[-1]/nchunk
		i2 = i1+chunk_size
		sub = tod[...,i1:i2]
		if sub.shape[-1] < 2: continue
		dtod = sub[...,1:]-sub[...,:-1]
		vs.append(np.mean(dtod**2,-1)/2)
	return np.median(vs,0)

def find_spikes(tod, nsigma=10, width=15, padding=7, noise=None):
	res = []
	if noise is None: noise = estimate_white_noise(tod)**0.5
	for di, d in enumerate(ftod):
		smooth = scipy.signal.medfilt(d, width)
		bad = np.abs(d-smooth) > noise[di]*nsigma
		bad = sampcut.from_mask(bad).widen(padding)
		res.append(bad)
	res = sampcut.stack(res)
	res.data.reshape(tod.shape[:-1])
	return res

def deglitch(tod, nsigma=10, width=15, padding=7, inplace=False):
	spikes = find_spikes(tod)
	return gapfill.gapfill_linear(tod, spikes, inplace=inplace)

def project(tod, basis, weight=1):
	rhs = basis.dot(np.conj((tod*weight)).T)
	div = basis.dot(np.conj((basis*weight)).T)
	amp = np.linalg.solve(div, rhs)
	return np.conj(amp).T.dot(basis)

def fit_common(tods, cuts=None, niter=None, overlap=None, clean_tod=False, weight=None):
	# for the given tods[ndet,nsamp], cuts (multirange[ndet,nsamp]) and az[nsamp],
	if not clean_tod: tods = tods.copy()
	if niter is None: niter = 3
	if weight is None:
		weight = np.full(len(tods), 1.0, dtype=tods.dtype)
	elif weight is "auto":
		weight = 1/estimate_white_noise(tods)
		weight /= np.mean(weight)
	# Output and work arrays
	res = tods[0]*0
	div = np.sum(tods*0+weight[:,None],0)
	for i in range(niter):
		# Overall logic: gapfill -> bin -> subtract -> loop
		if cuts is not None:
			gapfill.gapfill_linear(tods, cuts, overlap=overlap, inplace=True)
		delta = np.sum(tods*weight[:,None],0)
		delta /= div
		res += delta
		tods -= delta[None]
	return res

def fit_phase_flat(tods, az, daz=1*utils.arcmin, cuts=None, niter=None,
		overlap=None, clean_tod=False, weight=None):
	# for the given tods[ndet,nsamp], cuts (multirange[ndet,nsamp]) and az[nsamp],
	if not clean_tod: tods = tods.copy()
	if daz is None: daz = 1*utils.arcmin
	if niter is None: niter = 3
	if weight is None:
		weight = np.full(len(tods), 1.0, dtype=tods.dtype)
	elif weight is "auto":
		weight = 1/estimate_white_noise(tods)
		weight /= np.mean(weight)
	# Set up phase pixels
	amin = np.min(az)
	amax = np.max(az)
	naz = int((amax-amin)/daz)+1
	pflat = pmat.PmatPhaseFlat(az, amin, daz, naz)
	# Output and work arrays
	phase  = np.zeros((2,naz),tods.dtype)
	dphase = phase.copy()
	div   = phase.copy()
	# Precompute div
	pflat.backward(tods*0+weight[:,None], div, -1)
	div[div==0] = 1
	print((np.mean(div)))
	for i in range(niter):
		# Overall logic: gapfill -> bin -> subtract -> loop
		if cuts is not None:
			gapfill.gapfill_linear(tods, cuts, overlap=overlap, inplace=True)
		pflat.backward(tods*weight[:,None], dphase)
		dphase /= div
		phase += dphase
		pflat.forward(tods, -dphase)
	return phase


# Problem: Our basis is noisy - both white noise and 1/f noise, and so are
# our vectors. We don't want to propagate this noise into each individual
# detector when deprojecting. We can try to limit this by reducing the number
# of degrees of freedom in the basis vectors, for example by smoothing them,
# but that will reduce our ability to capture rapidly varying signals. The
# most important signal variations happen on scales below 20-40 Hz. That lets
# us reduce the variance by a factor of 5-10, which helps. But it only helps
# on small scales. On large scales the increase in noise would be the same
# as before.
#
# Can fit an amplitude separately for each frequency bin. At most frequencies
# there presumably wouldn't be much corelation, so one could skil subtracting it
# there.
#
# How about cross-correlating the basis vectors against the data, and only keeping
# the part with good correlation?

def fit_basis(tods, basis, highpass=50, cuts=None, clean_tod=True):
	if not clean_tod: tods = tods.copy()
	def hpass(a, n):
		f = fft.rfft(a)
		f[...,:n] = 0
		return fft.ifft(f,a.copy(),normalize=True)
	hdark = hpass(basis, highpass)
	for di in range(len(tods)):
		htod = hpass(tods[di], highpass)
		dark_tmp = hdark.copy()
		if cuts is not None:
			for ddi in range(len(hdark)):
				gapfill.gapfill(dark_tmp[ddi], cuts[di], inplace=True)
		fit = project(htod[None], dark_tmp)[0]
		# Subtract from original tod
		tods[di] -= fit
	return tods

def smooth_basis_fourier(ftod, fbasis, bsize=100, mincorr=0.1,
		nsigma=5, highpass=10, nmin=1):
	"""This function attemps to smooth out irrelevant fourier modes
	in a noisy set of basis fectors. It assumes the high frequencies
	are representative of the noise level, and removes parts of the
	bassi vectors that are too small relative to the noise level,
	and parts that do not correlate sufficiently with the tods that are
	passed in as the first argument."""
	fbasis = fbasis.copy()
	nbasis, nfreq = fbasis.shape
	nbin  = int((nfreq+bsize-1)/bsize)
	# Compute white noise level
	wbasis = np.var(fbasis[:,nfreq/2:],1)
	ngood = np.zeros(nbasis,dtype=int)
	for bi in range(nbin):
		r = [bi*bsize,(bi+1)*bsize]
		ft = ftod[:,r[0]:r[1]].copy()
		fd = fbasis[:,r[0]:r[1]].copy()
		# Compute mean normalized tod
		vtod = np.mean(ft*np.conj(ft),1).real
		vbasis= np.mean(fd*np.conj(fd),1).real
		ft /= vtod[:,None]**0.5
		fd /= vbasis[:,None]**0.5
		fmean = np.mean(ft,0)
		# Compute average correlation
		corr = np.mean(fmean*np.conj(fd),1).real
		# Reject a bin if it has too low S/N, or too low corr
		bad = (np.abs(corr) < mincorr) | (vbasis < wbasis*nsigma)
		fbasis[bad,r[0]:r[1]] = 0
		ngood[~bad] += 1
	fbasis[:,:highpass] = 0
	fbasis = fbasis[ngood>=nmin]
	return fbasis
