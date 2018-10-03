"""This module represents a model for point sources and similar localized objects
in maps. They are represented by their position, amplitude and ellipse. Only
gaussian profiles are supported for now. To add more profiles, PmatPtsrc needs
to be modified."""
import numpy as np
from . import utils, enmap

class SourceModel:
	def __init__(self, pos, amps=None, widths=None, phi=None, amp_lim=0):
		if amps is None:
			try: pos, amps, widths, phi = read_params(pos)
			except TypeError: pos, amps, widths, phi = expand_params(pos)
		# Expand amps to standard T,Q,U shape if necessary
		self.amps   = np.array(amps)[:3] # [T,Q,U]
		self.amps   = np.pad(self.amps, ((0,3-len(self.amps)),(0,0)), mode="constant")
		# Mask unworthy sources if necessary
		amp_lim   = np.atleast_1d(amp_lim).astype(np.float)
		amp_lim   = np.pad(amp_lim, ((0,3-len(amp_lim)),), mode="edge")
		mask      = np.any((np.abs(self.amps).T >= amp_lim.T).T,0)
		# Extract relevant part of each
		self.amps   = self.amps[:,mask]
		self.pos    = np.array(pos)[:,mask]
		self.widths = np.array(widths)[:,mask]
		self.phi    = np.array(phi)[mask]
		# Record which sources we ended up using
		self.inds = np.where(mask)[0]
	@property
	def nsrc(self): return self.pos.shape[1]
	@property
	def ibeam(self): return np.array([utils.compress_beam(w,p) for w,p in zip(self.widths.T, self.phi)]).T
	@property
	def params(self): return np.concatenate([self.pos, self.amps, self.ibeam])
	def __repr__(self): return "SourceModel(nsrc=%d)" % self.nsrc
	def draw(self, shape, wcs, window=False, nsigma=10, pad=False):
		m = enmap.zeros((3,)+shape[-2:], wcs)
		w = np.max(self.widths)*nsigma
		# Find all sources that enter even partially into our area
		ipos = np.array([
			enmap.sky2pix(shape, wcs, self.pos-w, corner=True, safe=True),
			enmap.sky2pix(shape, wcs, self.pos+w, corner=True, safe=True)]).astype(int)
		ipos = np.sort(ipos,0)
		ashape = np.array(shape[-2:])
		mask = np.all(ipos[1]>=0,0) & np.all(ipos[0]<ashape[:,None],0) & np.all(ipos[1]-ipos[0]<ashape[:,None],0)
		nmask= np.sum(mask)
		if nmask == 0: return m
		if pad:
			# We will include enough pixels on each side that we avoid
			# truncating any part of the source
			ipos2 = np.array([
				enmap.sky2pix(shape, wcs, self.pos[:,mask]-w*2, corner=True, safe=True),
				enmap.sky2pix(shape, wcs, self.pos[:,mask]+w*2, corner=True, safe=True)]).astype(int)
			npad  = np.maximum(0,[-np.min(ipos2,(0,2)),np.max(ipos2,(0,2))-ashape])
			m, mslice = enmap.pad(m, npad, return_slice=True)
		for i, (pos, amp, width, ibeam) in enumerate(zip(self.pos.T[mask], self.amps.T[mask], self.widths.T[mask], self.ibeam.T[mask])):
			# Find necessary bounding box
			sub = m.submap([pos-w,pos+w])
			if sub.size == 0: continue
			add_gauss(sub, pos, amp, ibeam)
		# Optinally apply window function
		if window: m = enmap.apply_window(m)
		if pad: m = m[mslice]
		return m

def add_gauss(m, pos, amp, ibeam):
	"""Helper function which draws a single gaussian on the enmap,
	modifiying m."""
	dx = m.posmap()-np.array(pos)[:,None,None]
	imat = np.array([[ibeam[0],ibeam[2]],[ibeam[2],ibeam[1]]])
	r2 = np.einsum("iyx,ij,jyx->yx",dx,imat,dx)
	m += amp[:,None,None]*np.exp(-0.5*r2)

def expand_params(params):
	pos   = params[:2]
	amps  = params[2:-3]
	widths= np.zeros([2,params.shape[1]])
	phi   = np.zeros([params.shape[1]])
	for i, ib in enumerate(params[-3:].T):
		widths[:,i], phi[i] = utils.expand_beam(ib)
	return pos, amps, widths, phi

def read_params(params):
	d = np.loadtxt(params).T
	return d[:2]*utils.degree, d[2:5], d[5:7]*utils.fwhm*utils.arcmin, d[7]*utils.degree

