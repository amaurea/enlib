"""This module provides an interface for inverse noise matrices.
Individual experiments can inherit from this - other functions
in enlib will work as long as this interface is followed.
For performance and memory reasons, the noise matrix
overwrites its input array."""
import numpy as np, enlib.fft, copy, enlib.slice, enlib.array_ops, h5py
import nmat_core_32, nmat_core_64

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
		core.nmat_covs(ft.T, self.get_ibins(tod.shape[1]).T, self.icovs.T/fft_norm)
		enlib.fft.irfft(ft, tod)
		return tod
	def white(self, tod):
		tod *= self.tdiag[:,None]
		return tod
	@property
	def ivar(self): return self.tdiag
	def get_ibins(self, n):
		nf = n/2+1
		ibins = (self.bins*nf/self.bins[-1,-1]).astype(np.int32)
		ibins[-1,-1] = nf
		return ibins
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
		return BinnedNmat(dets, bins, icovs)
	def __mul__(self, a):
		return NmatBinned(self.icovs/a, self.bins, self.dets)

class NmatDetvecs(NmatBinned):
	"""A binned noise matrix where the inverse covariance matrix is stored and
	used in a compressed form, as a set of eigenvectors and eigenvalues."""
	def __init__(self, D, V, E, bins, vbins, dets=None):
		"""Construct an NmatDetvecs.
		dets[ndet]:
		 the id of each detector described
		bins[nbin,2]:
		 the start and end point of each frequency bin in Hz
		D[nbin,ndet], vbins[nbin,2], V[nvecs,ndet], E[nvecs]
		 these specify the covariance in the bins, such that
		 cov[bin,d1,d2] = D[bin,d1]*delta(d1,d2)+sum(V[vs,d1]*V[vs,d2]*E[vs],vs=vbins[bin,0]:vbins[bin,1])
		That is, for each bin, cov = diag(D) + V.T.dot(np.diag(E)).dot(V).

		Note that D, V and E correspond to the normal covmat, *not* the inverse!
		"""
		self.D    = np.array(D)
		self.V    = np.array(V)
		self.E    = np.array(E)
		self.bins = np.array(bins)
		self.vbins= np.array(vbins)
		self.dets = np.array(dets) if dets is not None else np.arange(self.D.shape[1])
		# Compute corresponding parameters for the inverse
		self.iD, self.iV, self.iE = woodbury_invert(self.D, self.V, self.E, self.vbins)

		# Compute white noise approximation
		ticov = np.zeros([len(self.dets),len(self.dets)])
		for d,b,vb in zip(self.iD, self.bins, self.vbins):
			v, e = self.iV[vb[0]:vb[1]], self.iE[vb[0]:vb[1]]
			ticov += (np.diag(d) + v.T.dot(np.diag(e)).dot(v))*(b[1]-b[0])
		ticov /= np.sum(self.bins[:,1]-self.bins[:,0])

		self.tdiag = np.diag(ticov)
	@property
	def icovs(self):
		nbin, ndet = self.iD.shape
		res = np.empty([nbin,ndet,ndet])
		for bi,(d,vb) in enumerate(zip(self.iD, self.vbins)):
			v, e = self.iV[vb[0]:vb[1]], self.iE[vb[0]:vb[1]]
			res[bi] = np.diag(d) + v.T.dot(np.diag(e)).dot(v)
		return res
	def apply(self, tod):
		ft = enlib.fft.rfft(tod)
		fft_norm = tod.shape[1]
		core = get_core(tod.dtype)
		core.nmat_detvecs(ft.T, self.get_ibins(tod.shape[-1]).T, self.iD.T/fft_norm, self.iV.T, self.iE/fft_norm, self.vbins.T)
		enlib.fft.irfft(ft, tod)
		return tod
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		dets = res.dets[detslice]
		# Reduce sample rate if necessary
		step = np.abs(sampslice.step or 1)
		fmax = res.bins[-1,-1]/step
		mask = res.bins[:,0] < fmax
		bins, vbins = res.bins[mask], res.vbins[mask]
		bins[-1,-1] = fmax
		# Slice covs, not icovs
		return NmatDetvecs(res.D[mask][:,detslice], res.V[:,detslice], res.E, bins, vbins, dets)
	def __mul__(self, a):
		return NmatDetvecs(self.D/a, self.V, self.E/a, self.bins, self.vbins, self.dets)

def read_nmat(fname, group=None):
	if isinstance(fname, basestring):
		f = h5py.File(fname, "r")
	else:
		f = fname
	g = f[group] if group else f
	typ = np.array(g["type"])[...]
	if typ == "detvecs":
		return NmatDetvecs(g["D"], g["V"], g["E"], g["bins"], g["vbins"], g["dets"])
	elif typ == "binned":
		return NmatBinned(g["icovs"], g["bins"], g["dets"])
	else:
		raise IOError("Unrecognized noise matrix format %s" % typ)
	if isinstance(fname, basestring):
		f.close()

def write_nmat(fname, nmat, group=None):
	if isinstance(fname, basestring):
		f = h5py.File(fname, "w")
	else:
		f = fname
	prefix = group + "/" if group else ""
	if isinstance(nmat, NmatDetvecs):
		for k,v in [("type","detvecs"),("D",nmat.D),("V",nmat.V),("E",nmat.E),("bins",nmat.bins),("vbins",nmat.vbins),("dets",nmat.dets)]:
			f[prefix+k] = v
	elif isinstance(nmat, NmatBinned):
		for k,v in [("type","binned"),("icovs",nmat.icovs),("bins",nmat.bins),("dets",nmat.dets)]:
			f[prefix+k] = v
	if isinstance(fname, basestring):
		f.close()

def woodbury_invert(D, V, E, vbins=None):
	"""Given a compressed representation C = D + V'EV, compute a
	corresponding representation for inv(C) using the Woodbury
	formula."""
	if vbins is None:
		iD, iE  = 1./D, 1./E
		iD[~np.isfinite(iD)] = 0
		iE[~np.isfinite(iE)] = 0
		if len(iE) == 0: return iD, V, iE
		core, sign = sichol(np.diag(iE) + V.dot(np.diag(iD)).dot(V.T))
		iV   = (np.diag(iD).dot(V.T).dot(core)).T
		return iD, iV, np.zeros(len(E))-sign
	else:
		assert(D.ndim == 2)
		iD, iV, iE = D.copy(), V.copy(), E.copy()
		for b, vb in enumerate(vbins):
			iD[b], iV[vb[0]:vb[1]], iE[vb[0]:vb[1]] = woodbury_invert(D[b], V[vb[0]:vb[1]], E[vb[0]:vb[1]])
		return iD, iV, iE

def sichol(A):
	iA = np.linalg.inv(A)
	try:
		return np.linalg.cholesky(iA), 1
	except np.linalg.LinAlgError:
		return np.linalg.cholesky(-iA), -1
