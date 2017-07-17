import numpy as np, time, h5py
from scipy import signal
from enlib import config, fft, utils, gapfill, todops, pmat, rangelist

config.default("gfilter_jon_naz", 16, "The number of azimuth modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_nt",  10, "The number of time modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_nhwp", 0, "The number of hwp modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_niter", 3, "The number of time modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_phase", True, "Modify Jon's polynomial ground filter to use phase instead of azimuth.")

def filter_poly_jon(tod, az, weights=None, naz=None, nt=None, niter=None, cuts=None, hwp=None, nhwp=None, deslope=True, inplace=True, use_phase=None):
	"""Fix naz Legendre polynomials in az and nt other polynomials
	in t jointly. Then subtract the best fit from the data.
	The subtraction is inplace, so tod is modified. If naz or nt are
	negative, they are fit for, but not subtracted.
	NOTE: This function may leave tod nonperiodic.
	"""
	naz = config.get("gfilter_jon_naz", naz)
	nt  = config.get("gfilter_jon_nt", nt)
	nhwp= config.get("gfilter_jon_nhwp", nhwp)
	niter = config.get("gfilter_jon_niter", niter)
	use_phase = config.get("gfilter_jon_phase", use_phase)
	if not inplace: tod = tod.copy()
	do_gapfill = cuts is not None
	# No point in iterating if we aren't gapfilling
	if not do_gapfill: niter = 1
	if hwp is None or np.all(hwp==0): nhwp = 0
	naz, asign = np.abs(naz), np.sign(naz)
	nt,  tsign = np.abs(nt),  np.sign(nt)
	nhwp,hsign = np.abs(nhwp),np.sign(nhwp)
	d   = tod.reshape(-1,tod.shape[-1])
	if naz == 0 and nt == 0 and nhwp == 0: return tod
	# Build our set of basis functions. These are shared
	# across iterations.
	B = np.zeros([naz+nt+nhwp,d.shape[-1]],dtype=tod.dtype)
	if naz > 0:
		if not use_phase:
			# Build azimuth basis as polynomials
			x = utils.rescale(az,[-1,1])
			B[0] = x
			for i in range(1,naz): B[i] = B[i-1]*x
		else:
			x = build_phase(az)*np.pi
			for i in range(naz):
				j = i/2+1
				B[i] = np.cos(j*x) if i%2 == 0 else np.sin(j*x)
	if nt > 0:
		x = np.linspace(-1,1,d.shape[-1],endpoint=False)
		B[naz] = x
		for i in range(1,nt): B[naz+i] = B[naz+i-1]*x
	if nhwp > 0:
		# Use sin and cos to avoid discontinuities
		for i in range(nhwp):
			j = i/2+1
			x = np.cos(j*hwp) if i%2 == 0 else np.sin(j*hwp)
			B[naz+nt+i] = x
	for it in range(niter):
		if do_gapfill: gapfill.gapfill(d, cuts, inplace=True)
		# Solve for the best fit for each detector, [nbasis,ndet]
		# B[b,n], d[d,n], amps[b,d]
		if weights is None:
			try:
				amps = np.linalg.solve(B.dot(B.T),B.dot(d.T))
			except np.linalg.LinAlgError as e:
				print "LinAlgError in todfilter. Skipping"
				continue
		else:
			w = weights.reshape(-1,weights.shape[-1])
			amps = np.zeros([naz+nt+nhwp,d.shape[0]],dtype=tod.dtype)
			for di in range(len(tod)):
				try:
					amps[:,di] = np.linalg.solve((B*w[di]).dot(B.T),B.dot(w[di]*d[di]))
				except np.linalg.LinAlgError as e:
					print "LinAlgError in todfilter di %d. Skipping" % di
					continue
		# Subtract the best fit
		if asign > 0: d -= amps[:naz].T.dot(B[:naz])
		if tsign > 0: d -= amps[naz:naz+nt].T.dot(B[naz:naz+nt])
		if hsign > 0: d -= amps[naz+nt:naz+nt+nhwp].T.dot(B[naz+nt:naz+nt+nhwp])
	# Why was this necessary?
	if do_gapfill: gapfill.gapfill(d, cuts, inplace=True)
	if deslope: utils.deslope(tod, w=8, inplace=True)
	res = d.reshape(tod.shape)
	return res

def build_phase(az, smooth=3):
	""""Phase" is an angle that increases with az while az increases, but
	continues to increase as az falls."""
	mi,ma = utils.minmax(az)
	phase = (az-mi)/(ma-mi)
	falling = phase[1:]-phase[:-1] < 0
	falling = np.concatenate([falling[:1],falling])
	phase[falling] = 2-phase[falling]
	if smooth:
		phase = signal.medfilt(phase, 2*smooth+1)
	return phase

def deproject_vecs(tods, dark, nmode=50, cuts=None, deslope=True, inplace=True):
	"""Given a tod[ndet,nsamp] and a set of basis modes dark[nmode,nsamp], fit
	each tod in to the basis modes and subtract them from the tod. The fit
	ignores the lowest nmode fourier modes, and cut regions are approximately ignored."""
	if not inplace: tods=tods.copy()
	todops.fit_basis(tods, dark, highpass=nmode, cuts=cuts, clean_tod=True)
	if deslope: utils.deslope(tods, w=8, inplace=True)
	return tods

def deproject_vecs_smooth(tods, dark, nmode=50, cuts=None, deslope=True, inplace=True):
	if not inplace: tods=tods.copy()
	dark = dark.copy()
	ftod  = fft.rfft(tods)
	fdark = fft.rfft(dark)
	fdark = todops.smooth_basis_fourier(ftod, fdark)
	smooth= np.zeros((fdark.shape[0],dark.shape[1]),dtype=dark.dtype)
	fft.ifft(fdark, smooth, normalize=True)
	todops.fit_basis(tods, smooth, highpass=nmode, cuts=cuts, clean_tod=True)
	if deslope: utils.deslope(tods, w=8, inplace=True)

def filter_common_blockwise(tods, blocks, cuts=None, niter=None,
		deslope=True, inplace=True, weight="auto", nmin=5):
	# Loop over and filter each block
	#np.savetxt("comtod0.txt", tods[0])
	if not inplace: tods = tods.copy()
	for bi, block in enumerate(blocks):
		if len(block) < nmin: continue
		btod = np.ascontiguousarray(tods[block])
		bcut = None if cuts is None else cuts[block]
		common = todops.fit_common(btod, cuts=bcut, niter=niter, clean_tod=True, weight=weight)
		tods[block] = btod
		#if 0 in block: np.savetxt("comcom.txt", common)
	if deslope: utils.deslope(tods, w=8, inplace=True)
	#np.savetxt("comtod1.txt", tods[0])
	#1/0
	return tods

def filter_phase_blockwise(tods, blocks, az, daz=None, cuts=None, niter=None,
		deslope=True, inplace=True, weight="auto"):
	"""Given a tod[ndet,nsamp], fit for a common azimuth phase signal
	per block in blocks[nblock][dets], and subtract it. The binning size
	is given in arc minutes."""
	# Loop over and filter each block
	#np.savetxt("moo0.txt", tods[0])
	if not inplace: tods = tods.copy()
	for bi, block in enumerate(blocks):
		btod = np.ascontiguousarray(tods[block])
		bcut = None if cuts is None else cuts[block]
		phase = todops.fit_phase_flat(btod, az, daz=daz, cuts=bcut, niter=niter,
				clean_tod=True, weight=weight)
		tods[block] = btod
	if deslope: utils.deslope(tods, w=8, inplace=True)
	#np.savetxt("moo.txt", tods[0])
	#1/0
	return tods
