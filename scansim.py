from __future__ import division, print_function
import numpy as np, copy, warnings
from . import scan, coordinates, utils, nmat, pmat, array_ops, enmap, bunch, sampcut
from scipy import ndimage
warnings.filterwarnings("ignore")

def rand_srcs(box, nsrc, amp, fwhm, rand_fwhm=False):
	pos  = np.array([np.random.uniform(box[0,1],box[1,1],nsrc),np.random.uniform(box[0,0],box[1,0],nsrc)]).T
	amps = np.random.exponential(scale=amp, size=nsrc)
	amps *= 2*np.random.randint(low=0,high=2,size=nsrc)-1 # both sign sources for fun
	pos_angs = np.random.uniform(0, np.pi, nsrc)
	pos_fracs = np.random.uniform(0, 1, nsrc)
	pos_comps = np.zeros([nsrc,3])
	pos_comps[:,0] = 1
	pos_comps[:,1] = np.cos(2*pos_angs)*pos_fracs
	pos_comps[:,2] = np.sin(2*pos_angs)*pos_fracs
	amps = amps[:,None]*pos_comps
	if rand_fwhm:
		skew = 2
		ofwhm = np.random.exponential(scale=fwhm**(1.0/skew), size=nsrc)**skew
		amps *= ((fwhm/ofwhm)**1)[:,None]
	else:
		ofwhm = np.zeros([nsrc]) + fwhm
	return bunch.Bunch(pos=pos,amps=amps,beam=ofwhm/(8*np.log(2))**0.5)

def build_bins_linear(fmax, nbin):
	edges = np.linspace(0, fmax, nbin+1, endpoint=True)
	bins  = np.array([edges[:-1],edges[1:]]).T
	return bins

def white_noise(ndet, nsamp, sigma):
	bins  = build_bins_linear(1.0, 1)
	ebins = np.array([[0,0]])
	D     = np.zeros([1,ndet])+sigma**2
	V     = np.zeros([1,ndet])
	E     = np.zeros([1])
	return nmat.NmatDetvecs(D, V, E, bins, ebins)

def oneoverf_noise(ndet, nsamp, sigma, fknee=0.2, alpha=1):
	nbin  = 10000
	bins  = build_bins_linear(1.0, nbin)
	freq  = np.mean(bins,1)
	Nu    = np.empty([nbin,ndet])
	Nu[:,:] = ((1+(freq/fknee)**-alpha)*sigma**2)[:,None]
	#Nu[:,:] = ((0+(freq/fknee)**-alpha)*sigma**2)[:,None]
	ebins = np.zeros([nbin,2],dtype=int)
	return nmat.NmatDetvecs(Nu, np.zeros([1,ndet]), np.zeros([1]), bins, ebins)

def oneoverf_detcorr_noise(ndet, nper, nsamp, sigma, fknee=0.2, alpha=1, nmode=1):
	"""Simulate detector-correlated 1/f noise. The nmode argument specifies the number
	of correlated modes to simulate. Detectors in a group (group size given by nper)
	will always have 100% correlated correlated noise. The modes are relatively simple -
	simply fourier modes in the 1d group index."""
	nbin  = 1000
	bins  = build_bins_linear(1.0, nbin)
	freq  = np.mean(bins,1)
	Nu    = np.zeros([nbin,ndet])+sigma**2
	gi    = np.arange(ndet)//nper
	ngi   = ndet//nper+1
	modi  = np.arange(nmode)
	# The correlated modes [nmode,ndet]
	V     = np.cos(2*np.pi*modi[:,None]*gi[None,:]/ngi)
	# The power in these modes [nbin,nmode]
	E     = (freq[:,None]/fknee*(modi+1))**-alpha * sigma**2
	# Expand and flatten to [nvec,ndet] and [nvec], where nvec = nmode*nbin
	V     = np.repeat(V, nbin, 0)
	E     = E.reshape(-1)
	# The mapping info the flattened arrays
	ebins = build_bins_linear(nbin*nmode, nbin).astype(int)
	return nmat.NmatDetvecs(Nu, V, E, bins, ebins)

def scan_ceslike(nsamp, box, mjd0=55500, sys="hor", srate=100, azrate=1.5*utils.degree):
	t   = np.arange(nsamp,dtype=float)/srate
	maz = 0.5*(box[1,0]+box[0,0])
	waz = 0.5*(box[1,0]-box[0,0])
	boresight = np.zeros([nsamp,3])
	boresight[:,0] = t
	boresight[:,1] = maz + utils.triangle_wave(t*azrate, 4*waz)*waz
	boresight[:,2] = box[0,1]+(box[1,1]-box[0,1])*np.arange(nsamp)/nsamp
	phase = np.zeros([nsamp,2])
	return bunch.Bunch(boresight=boresight, hwp_phase=phase, sys=sys,mjd0=mjd0,site=coordinates.default_site)

def scan_grid(box, res, sys="equ", dir=0, margin=0):
	box[np.argmin(box,0)] += margin
	box[np.argmax(box,0)] -= margin
	n = np.round(np.asarray(box[1]-box[0])/res).astype(int)
	dec = np.linspace(box[0,0],box[1,0],n[0],endpoint=False) + res/2
	ra  = np.linspace(box[0,1],box[1,1],n[1],endpoint=False) + res/2
	if dir % 2 == 0:
		decra = np.empty([2,dec.size,ra.size])
		decra[0] = dec[:,None]
		decra[1] = ra [None,:]
	else:
		decra = np.empty([2,ra.size,dec.size])
		decra[0] = dec[None,:]
		decra[1] = ra [:,None]
	decra = decra.reshape(2,-1)
	t = np.arange(decra.shape[1])*1e3/decra.shape[1]
	boresight = np.empty([t.size,3])
	boresight[:,0] = t
	boresight[:,1:] = decra.T
	phase = np.zeros([t.size,2])
	return bunch.Bunch(boresight=boresight, hwp_phase=phase, sys=sys, mjd0=55500,site=coordinates.default_site)

def dets_scattered(nmul, nper=3, rad=0.5*np.pi/180, seed=0):
	ndet = nmul*nper
	np.random.seed(seed)
	offsets = np.repeat(np.random.uniform(size=[nmul,3])*rad, nper,0)
	offsets[:,0] = 0
	# T,Q,U sensitivity
	angles = np.arange(ndet)*np.pi/nmul
	comps     = np.zeros([ndet,3])
	comps[:,0] = 1
	comps[:,1] = np.cos(2*angles)
	comps[:,2] = np.sin(2*angles)
	return bunch.Bunch(comps=comps, offsets=offsets)

def dets_row(nmul, nper=3, rad=0.5*np.pi/180, dir=[1,1]):
	"""Simulate nmul groups of detectors in a row with a half-width of rad and a
	direction of dir, defaulting to a diagonal [1,1]."""
	ndet = nmul*nper
	dir  = np.array([0,dir[0],dir[1]])
	v    = rad*dir/np.sum(dir**2)**0.5
	offsets = np.repeat(np.linspace(-1,1,nmul), nper)[:,None]*v
	# T,Q,U sensitivity
	angles = np.arange(ndet)*np.pi/nmul
	comps  = np.zeros([ndet,3])
	comps[:,0] = 1
	comps[:,1] = np.cos(2*angles)
	comps[:,2] = np.sin(2*angles)
	return bunch.Bunch(comps=comps, offsets=offsets)

def nocut(ndet, nsamp):
	return sampcut.empty(ndet, nsamp)

class SimPlain(scan.Scan):
	def __init__(self, scanpattern, dets, noise, simsys="equ", cache=False, seed=0, noise_scale=1):
		# Set up the telescope
		self.boresight = scanpattern.boresight
		self.sys       = scanpattern.sys
		self.offsets   = dets.offsets
		self.comps     = dets.comps
		self.cut       = nocut(self.ndet,self.nsamp)
		self.mjd0      = scanpattern.mjd0
		# Set up the simulated signal properties
		self.noise = noise
		self.seed  = seed
		self.dets  = np.arange(len(self.comps))
		self.site  = scanpattern.site
		self.hwp_phase = scanpattern.hwp_phase
		self.noise_scale = noise_scale
		self.simsys = simsys
		self.id    = "sim"
	def get_samples(self):
		np.random.seed(self.seed)
		tod = np.zeros([self.ndet,self.nsamp])
		if self.noise_scale != 0:
			noise = np.random.standard_normal([self.ndet,self.nsamp])*self.noise_scale
			covs = array_ops.eigpow(self.noise.icovs, -0.5, axes=[-2,-1])
			N12  = nmat.NmatBinned(covs, self.noise.bins, self.noise.dets)
			N12.apply(noise)
			tod += noise
		return tod
	def get_model(self, point):
		return np.zeros((3,)+point.shape[:-1])
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		return res

class SimMap(SimPlain):
	def __init__(self, scanpattern, dets, map, noise, simsys="equ", cache=False, seed=0, noise_scale=1):
		SimPlain.__init__(self, scanpattern, dets, noise, simsys=simsys, cache=cache, seed=seed, noise_scale=noise_scale)
		self.map   = map.copy()
		self.pmat  = pmat.PmatMap(self, self.map, sys=simsys)
	def get_samples(self):
		tod = SimPlain.get_samples(self).astype(self.map.dtype)
		self.pmat.forward(tod, self.map, tmul=1)
		return tod
	def get_model(self, point):
		pix = self.map.sky2pix(point.T[::-1])
		return utils.interpol(self.map, pix, order=0).T

class SimSrcs(SimPlain):
	def __init__(self, scanpattern, dets, srcs, noise, simsys="equ", cache=False, seed=0, noise_scale=1, nsigma=4):
		SimPlain.__init__(self, scanpattern, dets, noise, simsys=simsys, cache=cache, seed=seed, noise_scale=noise_scale)
		self.srcs  = srcs
		self.nsigma = nsigma
		if cache: self._tod = None
	def get_samples(self):
		# Start with the noise
		if hasattr(self, "_tod") and self._tod is not None:
			return self._tod.copy()
		tod = SimPlain.get_samples(self)
		tod = tod.astype(np.float64)
		# And add the point sources
		for di in range(self.ndet):
			for i, (pos,amp,beam) in enumerate(zip(self.srcs.pos,self.srcs.amps,self.srcs.beam)):
				point = (self.boresight+self.offsets[di,None,:])[:,1:]
				point = coordinates.transform(self.sys, self.simsys, point.T, time=self.boresight[:,0]+self.mjd0, site=self.site).T
				r2 = np.sum((point-pos[None,:])**2,1)/beam**2
				I  = np.where(r2 < self.nsigma**2)[0]
				tod[di,I] += np.exp(-0.5*r2[I])*np.sum(amp*self.comps[di])
		if hasattr(self, "_tod"):
			self._tod = tod.copy()
		return tod
	def get_model(self, point):
		res = np.zeros([point.shape[0],self.srcs.amps.shape[1]])
		for i, (pos,amp,beam) in enumerate(zip(self.srcs.pos,self.srcs.amps,self.srcs.beam)):
			r2 = np.sum((point-pos[None,:])**2,1)/beam**2
			I  = np.where(r2 < 6**2)[0]
			res[I,:] += np.exp(-0.5*r2[I])[:,None]*amp[None,:]
		return res
