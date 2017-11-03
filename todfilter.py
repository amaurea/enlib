import numpy as np, time, h5py
from scipy import signal
from enlib import config, fft, utils, gapfill, todops, pmat

config.default("gfilter_jon_naz", 16, "The number of azimuth modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_nt",  10, "The number of time modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_nhwp", 0, "The number of hwp modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_niter", 3, "The number of time modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_phase", False, "Modify Jon's polynomial ground filter to use phase instead of azimuth.")

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
	d     = tod.reshape(-1,tod.shape[-1])
	nsamp = d.shape[-1]
	if naz == 0 and nt == 0 and nhwp == 0: return tod

	B = []
	# Set up our time basis
	if nt  > 0:
		B.append(np.full((1,nsamp), 1.0, d.dtype))
	if nt  > 1:
		t = np.linspace(-1,1,nsamp,endpoint=False)
		B.append(utils.build_legendre(t, nt-1))
	if naz > 0:
		if not use_phase:
			# Set up our azimuth basis
			B.append(utils.build_legendre(az, naz))
		else:
			# Set up phase basis. Vectors should be periodic
			# in phase to avoid discontinuities. cossin is good for this.
			phase = build_phase(az)*np.pi
			B.append(utils.build_cossin(phase, naz))
	if nhwp > 0:
		B.append(utils.build_cossin(hwp, nhwp))

	B = np.concatenate(B,0)

	for it in range(niter):
		if do_gapfill: gapfill.gapfill(d, cuts, inplace=True)
		# Solve for the best fit for each detector, [nbasis,ndet]
		# B[b,n], d[d,n], amps[b,d]
		if weights is None:
			try:
				amps = np.linalg.solve(B.dot(B.T),B.dot(d.T))
			except np.linalg.LinAlgError as e:
				print("LinAlgError in todfilter. Skipping")
				continue
		else:
			w = weights.reshape(-1,weights.shape[-1])
			amps = np.zeros([naz+nt+nhwp,d.shape[0]],dtype=tod.dtype)
			for di in range(len(tod)):
				try:
					amps[:,di] = np.linalg.solve((B*w[di]).dot(B.T),B.dot(w[di]*d[di]))
				except np.linalg.LinAlgError as e:
					print(("LinAlgError in todfilter di %d. Skipping" % di))
					continue
		# Subtract the best fit, but skip some basis functions if requested
		if tsign < 0: amps[:nt] = 0
		if asign < 0: amps[nt:nt+naz] = 0
		if hsign < 0: amps[nt+naz:nt+naz+nhwp] = 0
		d -= amps.T.dot(B)
	if do_gapfill: gapfill.gapfill(d, cuts, inplace=True)
	# This filtering might have casued the tod to become
	# non-continous, which would mess up future fourier transforms.
	# So it's safest to deslope here.
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
