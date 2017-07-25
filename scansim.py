import numpy as np, copy, warnings
from enlib import scan, coordinates, utils, nmat, pmat, array_ops, enmap, bunch
from enlib.bins import linbin
from scipy import ndimage
warnings.filterwarnings("ignore")

def rand_srcs(box, nsrc, amp, fwhm, rand_fwhm=False):
	pos  = np.array([np.random.uniform(box[0,1],box[1,1],nsrc),np.random.uniform(box[0,0],box[1,0],nsrc)]).T
	amps = np.random.exponential(scale=amp, size=nsrc)
	amps *= 2*np.random.randint(low=0,high=2,size=nsrc)-1 # both sign sources for fun
	pos_angs = np.random.uniform(0, np.pi, nsrc)
	pos_fracs = np.random.uniform(0, 1, nsrc)
	pos_comps = np.zeros([nsrc,4])
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

def white_noise(ndet, nsamp, sigma):
	bins  = linbin(1.0, 1)
	ebins = np.array([[0,0]])
	D     = np.zeros([1,ndet])+sigma**2
	V     = np.zeros([1,ndet])
	E     = np.zeros([1])
	return nmat.NmatDetvecs(D, V, E, bins, ebins)

def oneoverf_noise(ndet, nsamp, sigma, fknee=0.2, alpha=1):
	nbin  = 1000
	bins  = linbin(1.0, nbin)
	freq  = np.mean(bins,1)
	Nu    = np.empty([nbin,ndet])
	Nu[:,:] = ((1+(freq/fknee)**-alpha)*sigma**2)[:,None]
	#Nu[:,:] = ((0+(freq/fknee)**-alpha)*sigma**2)[:,None]
	ebins = np.zeros([nbin,2],dtype=int)
	return nmat.NmatDetvecs(Nu, np.zeros([1,ndet]), np.zeros([1]), bins, ebins)

def oneoverf_detcorr_noise(ndet, nsamp, sigma, fknee=0.2, alpha=1):
	# A single, atmospheric mode
	nbin  = 1000
	bins  = linbin(1.0, nbin)
	freq  = np.mean(bins,1)
	Nu    = np.zeros([nbin,ndet])+sigma**2
	E     = (freq/fknee)**-alpha * sigma**2
	V     = np.zeros([nbin,ndet])+1
	ebins = linbin(nbin,nbin)
	return nmat.NmatDetvecs(Nu, V, E, bins, ebins)

def scan_ceslike(nsamp, box, sys="equ", srate=100, azrate=0.123):
	t = np.arange(nsamp,dtype=float)/srate
	boresight = np.zeros([nsamp,3])
	boresight[:,0] = t
	boresight[:,1] = box[0,1]+(box[1,1]-box[0,1])*(1+np.cos(2*np.pi*t*azrate))/2
	boresight[:,2] = box[0,0]+(box[1,0]-box[0,0])*np.arange(nsamp)/nsamp
	return bunch.Bunch(boresight=boresight, sys=sys,mjd0=55500,site=scan.default_site)

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
	return bunch.Bunch(boresight=boresight, sys=sys, mjd0=55500,site=scan.default_site)

def dets_scattered(nmul, nper=3, rad=0.5*np.pi/180):
	ndet = nmul*nper
	offsets = np.repeat(np.random.uniform(size=[nmul,3])*rad, nper,0)
	offsets[:,0] = 0
	# T,Q,U sensitivity
	angles = np.arange(ndet)*np.pi/nmul
	comps     = np.zeros([ndet,4])
	comps[:,0] = 1
	comps[:,1] = np.cos(2*angles)
	comps[:,2] = np.sin(2*angles)
	return bunch.Bunch(comps=comps, offsets=offsets)

def nocut(ndet, nsamp):
	return sampcut.empty(ndet, nsamp)

class SimSrcs(scan.Scan):
	def __init__(self, scanpattern, dets, srcs, noise, simsys="equ", cache=False, seed=0, noise_scale=1, nsigma=4):
		# Set up the telescope
		self.boresight = scanpattern.boresight
		self.sys       = scanpattern.sys
		self.offsets   = dets.offsets
		self.comps     = dets.comps
		self.cut       = nocut(self.ndet,self.nsamp)
		self.mjd0      = scanpattern.mjd0
		# Set up the simulated signal properties
		self.srcs  = srcs
		self.noise = noise
		self.seed  = seed
		self.dets  = np.arange(len(self.comps))
		self.site  = scanpattern.site
		self.noise_scale = noise_scale
		self.simsys  = simsys
		self.nsigma = nsigma

		if cache: self._tod = None

	def get_samples(self):
		# Start with the noise
		if hasattr(self, "_tod") and self._tod is not None:
			return self._tod.copy()
		np.random.seed(self.seed)
		tod = np.zeros([self.ndet,self.nsamp]).astype(np.float32)
		if self.noise_scale != 0:
			tod  = np.random.standard_normal([self.ndet,self.nsamp]).astype(np.float32)*self.noise_scale
			covs = array_ops.eigpow(self.noise.icovs, -0.5, axes=[-2,-1])
			N12  = nmat.NmatBinned(covs, self.noise.bins, self.noise.dets)
			N12.apply(tod)
		else:
			tod[...] = 0
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

	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		return res

class SimMap(scan.Scan):
	def __init__(self, scanpattern, dets, map, noise, simsys="equ", cache=False, seed=0, noise_scale=1):
		# Set up the telescope
		self.boresight = scanpattern.boresight
		self.sys       = scanpattern.sys
		self.offsets   = dets.offsets
		self.comps     = dets.comps
		self.cut       = nocut(self.ndet,self.nsamp)
		self.mjd0      = scanpattern.mjd0
		# Set up the simulated signal properties
		self.map   = map.copy()
		self.noise = noise
		self.seed  = seed
		self.dets  = np.arange(len(self.comps))
		self.site  = scanpattern.site
		self.noise_scale = noise_scale
		self.simsys  = simsys
		self.pmat  = pmat.PmatMap(self, self.map, sys=simsys)
	def get_samples(self):
		np.random.seed(self.seed)
		tod = np.zeros([self.ndet,self.nsamp]).astype(self.map.dtype)
		self.pmat.forward(tod, self.map)
		if self.noise_scale:
			noise = np.random.standard_normal([self.ndet,self.nsamp]).astype(self.map.dtype)*self.noise_scale
			covs = array_ops.eigpow(self.noise.icovs, -0.5, axes=[-2,-1])
			N12  = nmat.NmatBinned(covs, self.noise.bins, self.noise.dets)
			N12.apply(noise)
			tod += noise
		return tod
	def get_model(self, point):
		pix = self.map.sky2pix(point.T[::-1])
		return utils.interpol(self.map, pix, order=0).T
