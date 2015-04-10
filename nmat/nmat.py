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
	def write(self, fname, group=None):
		fields = [("type","binned"),("icovs",self.icovs),("bins",self.bins),("dets",self.dets)]
		nmat_write_helper(fname, fields, group)

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
		tdiag /= np.sum(self.bins[:,1]-self.bins[:,0])

		self.tdiag = tdiag
	def calc_inverse(self):
		return woodbury_invert(self.D, self.V, self.E, self.ebins)
	@property
	def icovs(self): return expand_detvecs(self.iD, self.iE, self.iV, self.ebins)
	@property
	def covs(self): return expand_detvecs(self.D, self.E, self.V, self.ebins)
	def apply(self, tod):
		ft = enlib.fft.rfft(tod)
		fft_norm = tod.shape[1]
		core = get_core(tod.dtype)
		core.nmat_detvecs(ft.T, self.get_ibins(tod.shape[-1]).T, self.iD.T/fft_norm, self.iV.T, self.iE/fft_norm, self.ebins.T)
		enlib.fft.irfft(ft, tod)
		return tod
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		dets = res.dets[detslice]
		# Reduce sample rate if necessary
		step = np.abs(sampslice.step or 1)
		fmax = res.bins[-1,-1]/step
		mask = res.bins[:,0] < fmax
		bins, ebins = res.bins[mask], res.ebins[mask]
		bins[-1,-1] = fmax
		# Slice covs, not icovs
		return NmatDetvecs(res.D[mask][:,detslice], res.V[:,detslice], res.E, bins, ebins, dets)
	def __mul__(self, a):
		return NmatDetvecs(self.D/a, self.V, self.E/a, self.bins, self.ebins, self.dets)
	def write(self, fname, group=None):
		fields = [("type","detvecs"),("D",self.D),("V",self.V),("E",self.E),("bins",self.bins),("ebins",self.ebins),("dets",self.dets)]
		write_nmat_helper(fname, fields, group)

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
		return NmatSharedvecs(res.D[mask][:,detslice], res.V[:,detslice], res.E, bins, ebins, vbins, dets)
	def __mul__(self, a):
		return NmatDetvecs(self.D/a, self.V, self.E/a, self.bins, self.ebins, self.vbins, self.dets)
	def write(self, fname, group=None):
		fields = [("type","sharedvecs"),("D",self.D),("V",self.V),("E",self.E),("bins",self.bins),("ebins",self.ebins),("vbins",self.vbins),("dets",self.dets)]
		write_nmat_helper(fname, fields, group)

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
		ebins = g["ebins"].value if "ebins" in g else g["vbins"].value # compatibility with old format
		return NmatDetvecs(g["D"].value, g["V"].value, g["E"].value, g["bins"].value, ebins, g["dets"].value)
	elif typ == "sharedvecs":
		return NmatSharedvecs(g["D"].value, g["V"].value, g["E"].value, g["bins"].value, g["ebins"].value, g["vbins"].value, g["dets"].value)
	elif typ == "binned":
		return NmatBinned(g["icovs"], g["bins"], g["dets"])
	else:
		raise IOError("Unrecognized noise matrix format %s" % typ)
	if isinstance(fname, basestring):
		f.close()

def write_nmat(fname, nmat, group=None):
	"""Write noise matrix nmat to the named file, optionally
	under the named group."""
	nmat.write(fname, group=group)

def write_nmat_helper(fname, fields, group=None):
	if isinstance(fname, basestring):
		f = h5py.File(fname, "w")
	else:
		f = fname
	prefix = group + "/" if group else ""
	for k, v in fields:
		f[prefix+k] = v
	if isinstance(fname, basestring):
		f.close()

def woodbury_invert(D, V, E, ebins=None, vbins=None):
	"""Given a compressed representation C = D + V'EV, compute a
	corresponding representation for inv(C) using the Woodbury
	formula."""
	if ebins is None:
		iD, iE  = 1./D, 1./E
		iD[~np.isfinite(iD)] = 0
		iE[~np.isfinite(iE)] = 0
		if len(iE) == 0: return iD, V, iE
		arg = np.diag(iE) + (V*iD[None,:]).dot(V.T)
		core, sign = sichol(arg)
		iV = core.T.dot(V)*iD[None,:]
		return iD, iV, np.zeros(len(E))-sign
	else:
		assert(D.ndim == 2)
		iD, iE = D.copy(), E.copy()
		iV = np.zeros([E.shape[0],V.shape[1]])
		if vbins is None: vbins = ebins.copy()
		for b, (eb,vb) in enumerate(zip(ebins,vbins)):
			iD[b], iV[eb[0]:eb[1]], iE[eb[0]:eb[1]] = woodbury_invert(D[b], V[vb[0]:vb[1]], E[eb[0]:eb[1]])
		return iD, iV, iE

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
