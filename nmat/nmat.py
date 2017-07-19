"""This module provides an interface for inverse noise matrices.
Individual experiments can inherit from this - other functions
in enlib will work as long as this interface is followed.
For performance and memory reasons, the noise matrix
overwrites its input array."""
import numpy as np, enlib.fft, copy, enlib.slice, enlib.array_ops, enlib.utils, h5py
import nmat_core_32, nmat_core_64
from scipy.optimize import minimize

def get_core(dtype):
	if dtype == np.float32:
		return nmat_core_32.nmat_core
	else:
		return nmat_core_64.nmat_core

class NoiseMatrix:
	def apply(self, tod):
		"""Apply the full inverse noise matrix to tod. tod is overwritten,
		but also returned for convenience."""
		return tod
	def white(self, tod):
		"""Apply a fast, uncorrelated (white noise) approximation
		of the inverse noise matrix to tod. tod is overwritten, but also
		returned for convenience."""
		return tod
	def update(self, tod, srate):
		"""Returns a noise matrix fit to tod. Often this will be a no-op,
		but calling this method allows us to support delayed computation of
		the noise model."""
		return self
	def __getitem__(self, sel):
		"""Restrict noise matrix to a subset of detectors (first index)
		or a lower sampling rate (second slice). The last one must be
		a slice object, which must have empty start and stop values,
		and a positive step value."""
		return self
	@property
	def ivar(self): raise NotImplementedError
	def getitem_helper(self, sel):
		"""Expands sel to a detector and sample slice.
		The detector slice is straightforward. The sample slice
		may be less so. In fourier space, its effect is a rescaling
		and truncation, such that find2 = find1 * n2/n1,
		with find2_max = find1_max * n2/n1 / step, and n2 = stop-start."""
		if type(sel) != tuple: sel = (sel,)
		assert len(sel) < 3, "Too many indices in slice"
		detslice = sel[0] if len(sel) > 0 else slice(None)
		sampslice = sel[1] if len(sel) > 1 else slice(None)
		assert isinstance(sampslice,slice), "Sample part of slice must be slice object"
		res = copy.deepcopy(self)
		return res, detslice, sampslice

class NmatNull(NoiseMatrix):
	def __init__(self, dets=None):
		self.dets = dets
	def apply(self, tod):
		tod[:] = 0
		return tod
	@property
	def ivar(self): return np.zeros(len(self.dets))
	def white(self, tod): return self.apply(tod)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		if res.dets is not None: res.dets = res.dets[detslice]
		return res

class NmatBinned(NoiseMatrix):
	"""TOD noise matrices where power is assumed to be constant
	in a set of bins in frequency. Stores a covariance matrix for
	each such bin."""
	def __init__(self, icovs, bins, dets=None):
		"""Construct an NmatBinned given a list of detectors dets[ndet],
		a list of bins[nbin,{from,to}] in frequency, where bins[-1,-1]
		is assumed to be half the sampling rate, and icovs[nbin,ndet,ndet],
		the inverse covariance matrix for each bin."""
		# Public - should be provided by inheritors
		self.bins  = np.array(bins)
		self.icovs = np.array(icovs)
		self.dets  = np.array(dets) if dets is not None else np.arange(self.icovs.shape[1])
		# Private
		# The frequency-averaged inverse covariance matrix. Here the atmosphere will
		# be sub-dominant, so this matrix should be approximately diagonal
		ticov = np.sum((C*(b[1]-b[0]) for C,b in zip(self.icovs,self.bins)),0)
		ticov /= np.sum(self.bins[:,1]-self.bins[:,0])
		self.tdiag = np.diag(ticov)
	def apply(self, tod):
		ft = enlib.fft.rfft(tod)
		fft_norm = tod.shape[1]
		core = get_core(tod.dtype)
		core.nmat_covs(ft.T, get_ibins(self.bins, tod.shape[1]).T, self.icovs.T/fft_norm)
		enlib.fft.irfft(ft, tod, flags=['FFTW_ESTIMATE','FFTW_DESTROY_INPUT'])
		return tod
	def white(self, tod):
		tod *= self.tdiag[:,None]
		return tod
	@property
	def ivar(self): return self.tdiag
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		dets  = res.dets[detslice]
		step = np.abs(sampslice.step or 1)
		fmax = res.bins[-1,-1]/step
		mask = res.bins[:,0] < fmax
		bins, icovs = res.bins[mask], res.icovs[mask]
		bins[-1,-1] = fmax
		# Slice covs, not icovs
		covs  = enlib.array_ops.eigpow(icovs, -1, axes=[-2,-1])[mask][:,detslice][:,:,detslice]
		icovs = enlib.array_ops.eigpow(covs,  -1, axes=[-2,-1])
		# Downsampling changes the noise per sample
		return BinnedNmat(dets, bins, icovs*step)
	def __mul__(self, a):
		return NmatBinned(self.icovs/a, self.bins, self.dets)
	def export(self):
		return {"type":"binned","icovs":self.icovs,"bins":self.bins,"dets":self.dets}
	def write(self, fname):
		fields = self.export()
		nmat_write_helper(fname, fields)

class NmatDetvecs(NmatBinned):
	"""A binned noise matrix where the inverse covariance matrix is stored and
	used in a compressed form, as a set of eigenvectors and eigenvalues."""
	def __init__(self, D, V, E, bins, ebins, dets=None):
		"""Construct an NmatDetvecs.
		dets[ndet]:
		 the id of each detector described
		bins[nbin,2]:
		 the start and end point of each frequency bin in Hz
		D[nbin,ndet], ebins[nbin,2], V[nvecs,ndet], E[nvecs]
		 these specify the covariance in the bins, such that
		 cov[bin,d1,d2] = D[bin,d1]*delta(d1,d2)+sum(V[vs,d1]*V[vs,d2]*E[es],es=ebins[bin,0]:ebins[bin,1])
		That is, for each bin, cov = diag(D) + V.T.dot(np.diag(E)).dot(V).

		The units of D and E should be variance for each fourier mode, not sum
		for each fourier mode. I.e. a longer tod should not yield larger D and E
		values. Hence, they should have the same units as fft(d)**2/nsamp.

		Note that D, V and E correspond to the normal covmat, *not* the inverse!
		"""
		self.D    = np.ascontiguousarray(D)
		self.V    = np.ascontiguousarray(V)
		self.E    = np.ascontiguousarray(E)
		self.bins = np.ascontiguousarray(bins)
		self.ebins= np.ascontiguousarray(ebins)
		self.dets = np.ascontiguousarray(dets) if dets is not None else np.arange(self.D.shape[1])
		# Compute corresponding parameters for the inverse
		self.iD, self.iV, self.iE = self.calc_inverse()

		# Compute white noise approximation
		tdiag = np.zeros([len(self.dets)])
		for d,b,eb in zip(self.iD, self.bins, self.ebins):
			v, e = self.iV[eb[0]:eb[1]], self.iE[eb[0]:eb[1]]
			tdiag += (d + np.sum(v**2*e[:,None],0))*(b[1]-b[0])
			#tdiag += d*(b[1]-b[0])
		tdiag /= np.sum(self.bins[:,1]-self.bins[:,0])

		self.tdiag = tdiag
	def calc_inverse(self):
		return woodbury_invert(self.D, self.V, self.E, self.ebins)
	@property
	def icovs(self): return expand_detvecs(self.iD, self.iE, self.iV, self.ebins)
	@property
	def covs(self): return expand_detvecs(self.D, self.E, self.V, self.ebins)
	def apply(self, tod, inverse=False):
		ft = enlib.fft.rfft(tod)
		# Unit of noise model we apply:
		#  Assume we start from white noise with stddev s.
		#  FFT giving sum of N random numbers with dev s per mode:
		#  each mode will have dev s sqrt(N).
		#  Divide by sqrt(N) before estimating noise: each mode has dev s.
		#  So unit of iN is 1/s^2.
		#  We apply as ifft(iN fft(d))/N, which for white simlifies to ifft(fft(d))/N /s^2.
		#  Since our ifft/N is the real, normalized ifft, so this simplifies to d/s^2. Good.
		# What happens when we downgrade by factor D:
		#  The noise variance per sample in the TOD should go down by factor D, which means
		#  that we want iN to multiply d by D/s^2
		#  We don't do that here.
		#  That means that our iN is D times too small, and hence RHS is D times too small.
		#  The same applies to div, so bin should come out OK. And the unit of iN does not
		#  really matter for the rest of the mapmaker.
		# To summarize, iN, RHS and div are all D times too small because we don't properly
		# rescale the noise when downsampling.
		# FIXED: I now scale D and E when downsampling.
		self.apply_ft(ft, tod.shape[-1], tod.dtype, inverse=inverse)
		enlib.fft.irfft(ft, tod, flags=['FFTW_ESTIMATE','FFTW_DESTROY_INPUT'])
		return tod
	def apply_ft(self, ft, nsamp, dtype, inverse=False):
		fft_norm = nsamp
		core = get_core(dtype)
		if not inverse:
			core.nmat_detvecs(ft.T, get_ibins(self.bins, nsamp).T, self.iD.T/fft_norm, self.iV.T, self.iE/fft_norm, self.ebins.T)
		else:
			core.nmat_detvecs(ft.T, get_ibins(self.bins, nsamp).T, self.D.T/fft_norm, self.V.T, self.E/fft_norm, self.ebins.T)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		dets = res.dets[detslice]
		# Reduce sample rate if necessary
		step = np.abs(sampslice.step or 1)
		fmax = res.bins[-1,-1]/step
		mask = res.bins[:,0] < fmax
		bins, ebins = res.bins[mask], res.ebins[mask]
		bins[-1,-1] = fmax
		# Slice covs, not icovs. Downsampling changes the noise per sample, which is why we divide
		# the variances here.
		return NmatDetvecs(res.D[mask][:,detslice]/step, res.V[:,detslice], res.E/step, bins, ebins, dets)
	def __mul__(self, a):
		return NmatDetvecs(self.D/a, self.V, self.E/a, self.bins, self.ebins, self.dets)
	def export(self):
		return {"type":"detvecs","D":self.D,"V":self.V,"E":self.E,"bins":self.bins,"ebins":self.ebins,"dets":self.dets}
	def write(self, fname):
		fields = self.export()
		write_nmat_helper(fname, fields)

class NmatSharedvecs(NmatDetvecs):
	"""A binned noise matrix where the inverse covariance matrix is stored and
	used in a compressed form, as a set of eigenvectors and eigenvalues. Compared
	to NmatDetvecs, this format is significantly more compressed for some noise
	patterns, by allowing eigenvectors to be shared between frequency bins.

	This extra compression does not propagate to the inverse quantities that are used
	internally, however, so this is only relevant for storing these matrices on disk.
	"""
	def __init__(self, D, V, E, bins, ebins, vbins, dets=None):
		"""Construct an NmatDetvecs.
		dets[ndet]:
		 the id of each detector described
		bins[nbin,2]:
		 the start and end point of each frequency bin in Hz
		D[nbin,ndet], ebins[nbin,2], vbins[nbin,2], V[nvecs,ndet], E[neigs]
		 these specify the covariance in the bins, such that
		 cov[bin,d1,d2] = D[bin,d1]*delta(d1,d2)+sum(V[vs,d1]*V[vs,d2]*E[es],es=ebins[bin,0]:ebins[bin,1],vs=vbins[bin,0]:vbins[bin,1]])
		That is, for each bin, cov = diag(D) + V.T.dot(np.diag(E)).dot(V).

		Note that D, V and E correspond to the normal covmat, *not* the inverse!
		"""
		self.vbins = np.ascontiguousarray(vbins)
		NmatDetvecs.__init__(self, D, V, E, bins, ebins, dets=dets)
	def calc_inverse(self):
		return woodbury_invert(self.D, self.V, self.E, self.ebins, self.vbins)
	@property
	def covs(self): return expand_detvecs(self.D, self.E, self.V, self.ebins, self.vbins)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		dets = res.dets[detslice]
		# Reduce sample rate if necessary
		step = np.abs(sampslice.step or 1)
		fmax = res.bins[-1,-1]/step
		mask = res.bins[:,0] < fmax
		bins, ebins, vbins = res.bins[mask], res.ebins[mask], res.vbins[mask]
		bins[-1,-1] = fmax
		# Slice covs, not icovs
		return NmatSharedvecs(res.D[mask][:,detslice]/step, res.V[:,detslice], res.E/step, bins, ebins, vbins, dets)
	def __mul__(self, a):
		return NmatDetvecs(self.D/a, self.V, self.E/a, self.bins, self.ebins, self.vbins, self.dets)
	def export(self):
		return {"type":"sharedvecs","D":self.D,"V":self.V,"E":self.E,"bins":self.bins,"ebins":self.ebins,"vbins":self.vbins,"dets":self.dets}
	def write(self, fname):
		fields = self.export()
		write_nmat_helper(fname, fields)

# Plans for new nmat for HWP data and filter magrinalization.
# 1. Increased variance resoultion. Use much higher resolution
#    for the variance than for the correlation. Model will ve
#    cov = sqrt(var)*corr*sqrt(var)
# 2. Multi-1/f binning. Assume HWP is spinning at approximately constant
#    rate fh. The sky signal and polarized atmosphere will be modulated at
#    4fh, but since they are not constant they will have a spread around
#    this. Much like we have a 1/f behavior around 0Hz, we should expect
#    this around 4fh also. So we should have the same kind of increased
#    resolution here.
# 3. HWP emission notch filter. The HWP emission is modulated at multiples
#    of f. Since the rotation speed is not quite constant, we will need a
#    small delta-f width comparable to the rotation speed variation, which
#    should be at the <0.1% level.
# 4. Azimuth notch filter. We scan at a very regular pattern in az, so an
#    azimuth filter can be expressed as a time-domain fourier space filter.
#    If we do this, then we can compensate for this filter by using notches
#    in the noise model.
# 5. Noise spike notch filter. As before.
#
# However, many of these are affected by not having a period that matches
# the TOD period. How much are the spikes blurred in this case?
# int_0^T sin(pi*i*t/T) sin(2*pi*f) dt

# exp(ix)exp(iy) = exp(i(x+y)) = cos(x+y)+isin(x+y)
# = cos(x)cos(y) - sin(x)sin(y) +i(cos(x)sin(y)+sin(x)cos(y))
# so cos(x+y) - cos(x-y) = -sin(x)sin(y) + sin(x)sin(-y) = -2sin(x)sin(y)
# So sin(x)sin(y) = 0.5*(cos(x-y)-cos(x+y))
# So we have
# 0.5 int_0^T (cos(pi*i*t/T-2*pi*f*t)-cos(pi*i*t/T+2*pi*f*t)) dt
# 0.5 [-sin(pi*i*t/T-2*pi*f*t)+sin(pi*i*t/T+2*pi*f*t)]_0^T
# 0.5 [-sin(pi*i-2*pi*f*T)+sin(pi*i+2*pi*f*T)] 
# i even: sin(2*pi*f*T)
# i odd: -sin(2*pi*f*T)
# So power is in ancy case sin(2*pi*f*T)**2.

class NmatScaled(NoiseMatrix):
	"""A noise matrix similar to detvecs, but where the variance is binned in smaller
	bins than the correlation."""
	def __init__(self, scale, bins, nmat):
		"""Construct an NmatMultires, which has different resolution for the
		variances and correlation structure.
		
		scale[ndet,nsbin]: high-resolution *rms* scaling. So the square of this will be applied to the covariance.
		bins [nvbin,2]: from-to for each bin in arbitrary units. Will be scaled so that whole spec is covered.
		nmat: NoiseMatrix to apply scaling to.
		"""
		self.nmat      = nmat
		self.scale     = np.ascontiguousarray(scale)
		self.inv_scale = 1/self.scale
		self.bins      = np.ascontiguousarray(bins)

		# Get the white noise approximation. This is a bit difficult due
		# to the different binning the scaling uses. But it only needs to
		# be an approximation anyway. So we will treat the scaling and
		# the rest as separable.
		bsize = self.bins[:,1]-self.bins[:,0]
		avg_iS2 = (np.sum(self.inv_scale**2 * bsize,1)/np.sum(bsize))
		self.tdiag = self.nmat.tdiag * avg_iS2
	def apply(self, tod):
		ft = enlib.fft.rfft(tod)
		self.apply_ft(ft, tod.nsamp, tod.dtype)
		enlib.fft.irfft(ft, tod, flags=['FFTW_ESTIMATE','FFTW_DESTROY_INPUT'])
		return tod
	def apply_ft(self, ft, nsamp, dtype):
		ibins= get_ibins(self.bins, nsamp)
		get_core(dtype).nmat_uncorr(ft.T, ibins.T, self.inv_scale.T)
		self.nmat.apply_ft(ft, nsamp, dtype)
		get_core(dtype).nmat_uncorr(ft.T, ibins.T, self.inv_scale.T)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		step = np.abs(sampslice.step or 1)
		# First slice our internal noise matrix. But undo the
		# step scaling, as the overall variance is our responsibility.
		nmat = res.nmat[sel] * step**-1
		# Then handle our part
		fmax = res.bins[-1,-1]/step
		mask = res.bins[:,0] < fmax
		bins = res.bins[mask]
		bins[-1,-1] = fmax
		scale= res.scale[detslice,mask]
		return NmatScaled(scale/step**0.5, bins, nmat)
	def __mul__(self, a):
		return NmatScaled(self.scale*a**0.5, self.bins, self.nmat)
	def export(self):
		return {"type":"scaled", "bins":self.bins, "scale":self.scale, "nmat": self.nmat.export()}
	def write(self, fname):
		fields = self.export()
		write_nmat_helper(fname, fields)

#class NmatScaled2(NoiseMatrix):
#	"""A noise matrix similar to detvecs, but where the variance is binned in smaller
#	bins than the correlation."""
#	def __init__(self, vars, freqs, nmat):
#		"""Construct an NmatMultires, which has different resolution for the
#		variances and correlation structure.
#
#		vars[nvar,ndet]: variance model. Will be linearly interpolated
#		freqs[nvar]:     freq indices at which the variance model is specified
#		                 will be rescaled such that freqs[-1] = nfreq in tod
#		nmat: NoiseMatrix of scaled data
#		"""
#		self.nmat  = nmat
#		self.freqs = np.ascontiguouarray(freqs)
#		self.vars  = np.ascontiguousarray(vars)
#
#		# Get the white noise approximation. This is a bit difficult due
#		# to the different binning the scaling uses. But it only needs to
#		# be an approximation anyway. So we will treat the scaling and
#		# the rest as separable.
#		bins  = utils.edges2bins(self.freqs)
#		bsize = bins[:,1]-bins[:,0]
#		avg_ivar = np.sum(1/self.vars[:-1]*bsize[:,None],0)/np.sum(bsize)
#		self.tdiag = self.nmat.tdiag * avg_ivar
#	def apply(self, tod):
#		ft = enlib.fft.rfft(tod)
#		self.apply_ft(ft, tod.nsamp, tod.dtype)
#		enlib.fft.irfft(ft, tod, flags=['FFTW_ESTIMATE','FFTW_DESTROY_INPUT'])
#		return tod
#	def apply_ft(self, ft, nsamp, dtype):
#		ifreqs= get_ifreqs(self.freqs, nsamp)
#		get_core(dtype).scale_lin(ft.T, ifreqs, self.vars.T, -0.5)
#		self.nmat.apply_ft(ft, nsamp, dtype)
#		get_core(dtype).scale_lin(ft.T, ifreqs, self.vars.T, -0.5)
#	def __getitem__(self, sel):
#		res, detslice, sampslice = self.getitem_helper(sel)
#		step = np.abs(sampslice.step or 1)
#		# Undo the internal scaling in nmat: We want to handle
#		# that using scale itself. A common use case of this
#		# class will be to let scale be rms values and nmat
#		# a correlation matrix. We want to preserve that
#		# distinction.
#		nmat = res.nmat[sel] * step**-1
#		fmax = res.freqs[-1]/step
#		# We have two frequency binnings now. First do the high-res var binning
#		mask  = res.freqs < fmax
#		freqs = res.freqs[mask]
#		freqs[-1] = fmax
#		return NmatScaled2(res.vars/step**0.5, freqs, nmat)
#	def __mul__(self, a):
#		return NmatScaled2(self.vars*a, self.freqs, self.nmat)
#	def export(self):
#		return {"type":"scaled2", "freqs":self.freqs, "vars":self.vars, "nmat": self.nmat.export()}
#	def write(self, fname):
#		fields = self.export()
#		write_nmat_helper(fname, fields)


def read_nmat(fname, group=None):
	"""Read a noise matrix from file, optionally from the named group
	in the file."""
	if isinstance(fname, basestring):
		f = h5py.File(fname, "r")
	else:
		f = fname
	g = f[group] if group else f
	typ = np.array(g["type"])[...]
	if typ == "detvecs":
		return NmatDetvecs(g["D"].value, g["V"].value, g["E"].value, g["bins"].value, g["ebins"].value, g["dets"].value)
	elif typ == "sharedvecs":
		return NmatSharedvecs(g["D"].value, g["V"].value, g["E"].value, g["bins"].value, g["ebins"].value, g["vbins"].value, g["dets"].value)
	elif typ == "binned":
		return NmatBinned(g["icovs"], g["bins"], g["dets"])
	else:
		raise IOError("Unrecognized noise matrix format %s" % typ)
	if isinstance(fname, basestring):
		f.close()

def write_nmat(fname, nmat):
	"""Write noise matrix nmat to the named file"""
	nmat.write(fname)

def write_nmat_helper(fname, fields, prefix=""):
	if isinstance(fname, basestring):
		f = h5py.File(fname, "w")
	else:
		f = fname
	for k, v in fields.iteritems():
		if isinstance(v, dict):
			write_nmat_helper(f, v, prefix+k+"/")
		else:
			f[prefix+k] = v
	if isinstance(fname, basestring):
		f.close()

def woodbury_invert(D, V, E, ebins=None, vbins=None):
	"""Given a compressed representation C = D + V'EV, compute a
	corresponding representation for inv(C) using the Woodbury
	formula."""
	if ebins is None: return woodbury_invert_single(D,V,E)
	assert(D.ndim == 2)
	iD, iE = D.copy(), E.copy()
	iV = np.zeros([E.shape[0],V.shape[1]])
	if vbins is None: vbins = ebins.copy()
	for b, (eb,vb) in enumerate(zip(ebins,vbins)):
		iD[b], iV[eb[0]:eb[1]], iE[eb[0]:eb[1]] = woodbury_invert_single(D[b], V[vb[0]:vb[1]], E[eb[0]:eb[1]])
	return iD, iV, iE

def woodbury_invert_single(D, V, E):
	"""Given a compressed representation C = D + V'EV, compute a
	corresponding representation for inv(C) using the Woodbury
	formula."""
	iD, iE  = 1./D, 1./E
	iD[~np.isfinite(iD)] = 0
	iE[~np.isfinite(iE)] = 0
	if len(iE) == 0: return iD, V, iE
	arg = np.diag(iE) + (V*iD[None,:]).dot(V.T)
	core, sign = sichol(arg)
	iV = core.T.dot(V)*iD[None,:]
	return iD, iV, np.zeros(len(E))-sign

def sichol(A):
	iA = np.linalg.inv(A)
	try:
		return np.linalg.cholesky(iA), 1
	except np.linalg.LinAlgError:
		return np.linalg.cholesky(-iA), -1

def expand_detvecs(D, E, V, ebins, vbins=None):
	nbin, ndet = D.shape
	res = np.empty([nbin,ndet,ndet])
	if vbins is None: vbins = ebins
	for bi,(d,eb,vb) in enumerate(zip(D, ebins, vbins)):
		v, e = V[vb[0]:vb[1]], E[eb[0]:eb[1]]
		res[bi] = np.diag(d) + (v.T*e[None,:]).dot(v)
	return res

def decomp_DVEV(cov, nmax=15, mineig=0, maxeval=0, tol=1e-2, _mode_ratios=False):
	"""Decompose covariance matrix cov[n,n] into
	D[n,n] + V[n,m] E[m,m] V'[m,n], where D and E
	are diagonal and positive and V is orthogonal.
	The number of columns in V will be nmax. If
	mineig is specified, then modes with lower
	amplitude than mineig*(max(E),max(sum(D**2)))
	will be pruned. D,E,V are then recomputed from
	scratch with this lower number of modes.

	Returns D, E, V."""
	if nmax == 0: return np.diag(cov), np.zeros([0]), np.zeros([len(cov),0])
	if mineig > 0:
		# If mineig is specified, then we will automatically trim the number
		# of modes to those larger than mineig times the largest mode.
		ratios = decomp_DVEV(cov, nmax, mineig=0, _mode_ratios=True)
		nbig   = np.sum(ratios>mineig)
		return decomp_DVEV(cov, nbig, mineig=0)
	# We will work on the correlation matrix, as that gives all row and cols
	# equal weight.
	C, std = enlib.utils.cov2corr(cov)
	Q = enlib.utils.eigsort(C, nmax=nmax, merged=True)
	def dvev_chisq(x, shape, esc):
		if np.any(~np.isfinite(x)): return np.inf
		Q = x.reshape(shape)
		D = np.diag(C)-np.einsum("ia,ia->i",Q,Q)
		if np.any(D<=0): return np.inf
		Ce = Q.dot(Q.T)
		R = enlib.utils.nodiag(C-Ce)
		chi = np.sum(R**2)
		esc(chi, x)
		return np.sum(R**2)
	def dvev_jac(x, shape, esc):
		Q = x.reshape(shape)
		Ce = Q.dot(Q.T)
		R = enlib.utils.nodiag(C-Ce)
		dchi = -4*R.T.dot(Q)
		esc()
		return dchi.reshape(-1)
	try:
		sol = minimize(dvev_chisq, Q.reshape(-1), method="newton-cg", jac=dvev_jac, tol=tol, args=(Q.shape,MinimizeEscape(maxeval)))
		Q = sol.x.reshape(Q.shape)
	except MinimizeEscape as e:
		Q = e.bval.reshape(Q.shape)
	if _mode_ratios:
		# Helper mode: Only return the relative contribution of each mode
		e,v = enlib.utils.eigsort(Q.dot(Q.T), nmax=nmax)
		d = np.diag(C)-np.einsum("ia,ia->i",Q,Q)
		scale = max(np.max(e), np.sum(d**2))
		return e/scale
	# Rescale from corr to cov
	Q *= std[:,None]
	# And split into VEV
	E, V = enlib.utils.eigsort(Q.dot(Q.T), nmax=nmax)
	D  = np.diag(cov)-np.einsum("ia,a,ia->i",V,E,V)
	return D,E,V

class MinimizeEscape:
	def __init__(self, maxeval):
		self.i, self.n = 0, maxeval
		self.bchi, self.bval = np.inf, None
	def __call__(self, chi=None, val=None):
		if chi is not None and chi < self.bchi:
			self.bchi = chi
			self.bval = np.array(val)
		self.i += 1
		if self.i > self.n: raise self

def apply_window(tod, width, inverse=False):
	"""Apply tapering window on width samples at each side of the tod.
	For example, apply_window(tod, 5*srate) would apply a 5-second window
	at each edge of the tod."""
	width = int(width)
	if width == 0: return
	core = get_core(tod.dtype)
	core.apply_window(tod.T, -width if inverse else width)

def get_ibins(bins, n):
	nf = n/2+1
	ibins = (bins*nf/bins[-1,-1]).astype(np.int32)
	ibins[-1,-1] = nf
	return ibins

def get_ifreqs(freqs, n):
	nf = n/2+1
	res = (freqs*nf/freqs[-1]).astype(np.int32)
	res[-1] = nf
	return res
