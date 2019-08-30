import numpy as np, time, os, sys, healpy
from enlib import utils
with utils.nowarn(): import h5py
from scipy import ndimage, stats, spatial, integrate, optimize
from enlib import enmap, utils, mpi, curvedsky, bunch, parallax, cython, ephemeris, statdist
from enlib import nmat, pmat, sampcut, fft
from pixell import sharp

def displace_pos(pos, earth_pos, r, off):
	dec, ra = np.array(pos)
	cdec = np.cos(dec)
	dec, ra = dec+off[0], ra+off[1]/cdec
	x    = cdec*np.cos(ra)*r - earth_pos[0]
	y    = cdec*np.sin(ra)*r - earth_pos[1]
	z    = np.sin(dec)    *r - earth_pos[2]
	r_hor = np.sqrt(x**2+y**2)
	odec = np.arctan2(z,r_hor)
	ora  = np.arctan2(y,x)
	opos = np.array([odec,ora])
	return opos

def add_delta(map, fpix, amp=1, r=32):
	# Add a delta function at the given pixel location in the map,
	# supporting non-integer pixel positions
	ipix = np.floor(fpix).astype(int)
	dpix = fpix-ipix
	pixbox = np.array([ipix-r,ipix+r+1])
	sub    = map.extract_pixbox(pixbox)
	sub[r,r] = amp
	sub[:]   = fft.shift(sub, dpix)
	map.insert_at(pixbox, sub, op=np.add)
	return map

def calc_beam_area(beam_profile):
	r, b = beam_profile
	return integrate.simps(2*np.pi*r*b,r)

def merge_nearby(srcs, rlim=2*utils.arcmin):
	"""given source parameters which might contain duplicates, detect these duplicates
	and merge them to produce a single catalog with no duplicates. sources are considered
	duplicates if they are within rlim of each other."""
	pos    = np.array([srcs[:,1]*np.cos(srcs[:,0]), srcs[:,0]]).T
	tree   = spatial.cKDTree(pos)
	groups = tree.query_ball_tree(tree, rlim)
	done   = np.zeros(len(srcs),bool)
	ocat   = []
	for gi, group in enumerate(groups):
		# remove everything that's done
		group = np.array(group)
		group = group[~done[group]]
		if len(group) == 0: continue
		# Let amplitude be the max in the group, and the position be the
		# amp-weighted mean
		gsrcs  = srcs[group]
		osrc   = gsrcs[0]*0
		weight = gsrcs[:,2]
		best   = np.argmax(np.abs(weight))
		osrc[:2] = np.sum(gsrcs[:,:2]*weight[:,None],0)/np.sum(weight)
		osrc[2:] = gsrcs[best,2:]
		done[group] = True
		ocat.append(osrc)
	ocat = np.array(ocat)
	return ocat

def cut_bright_srcs(scan, srcs, alim_include=1e4, alim_size=10):
	"""Cut sources in srcs that are brighter than alim in uK"""
	srcs_cut = srcs[srcs[:,2]>alim_include]
	if len(srcs) == 0: return scan
	tod  = np.zeros((scan.ndet,scan.nsamp), np.float32)
	psrc = pmat.PmatPtsrc(scan, srcs_cut)
	psrc.forward(tod, srcs)
	cut  = sampcut.from_mask(tod > alim_size)
	scan.cut *= cut
	#scan.cut_noiseest *= cut
	# Ensure that our bright sources get gapfilled before anything else
	scan.d.cut_basic *= cut
	return scan

class NmatWindowed(nmat.NoiseMatrix):
	def __init__(self, nmat_inner, windows):
		self.inner  = nmat_inner
		self.windows = windows
	def apply(self, tod, white=False):
		for window in self.windows: window(tod)
		if not white: self.inner.apply(tod)
		else:         self.inner.white(tod)
		for window in self.windows[::-1]: window(tod)
		return tod
	def white(self, tod): return self.apply(tod, white=True)

class PsrcSimple:
	def __init__(self, scan, srcparam):
		self.inner = pmat.PmatPtsrc(scan, srcparam)
		self.srcparam = srcparam
	def forward(self, tod, amps, tmul=None, pmul=None):
		params = self.srcparam.copy()
		params[:,2:5] = 0; params[:,2] = amps
		self.inner.forward(tod, params, tmul=tmul, pmul=pmul)
		return tod
	def backward(self, tod, tmul=None):
		params = self.srcparam.copy()
		self.inner.backward(tod, params, pmul=0, tmul=tmul)
		return params[:,2]

def choose_corr_points(shape, wcs, spacing):
	# Set up the points where we will measure the correlation matrix
	box       = np.sort(enmap.box(shape, wcs),0)
	dstep     = spacing
	corr_pos = []
	for dec in np.arange(box[0,0]+dstep/2.0, box[1,0], dstep):
		astep = spacing/np.cos(dec)
		for ra in np.arange(box[0,1]+astep/2.0, box[1,1], astep):
			corr_pos.append([dec,ra])
	corr_pos = np.array(corr_pos)
	return corr_pos

def defmean(arr, defval=0):
	return np.mean(arr) if len(arr) > 0 else defval

def apply_beam_fft(map, bl):
	l    = map.modlmap()
	bval = np.interp(l, np.arange(len(bl)), bl, right=0)
	return enmap.ifft(enmap.fft(map)*bval).real

def apply_beam_sht(map, bl, tol=1e-5):
	lmax = np.where(bl/np.max(bl) > tol)[0][-1]
	ainfo= sharp.alm_info(lmax)
	alm  = curvedsky.map2alm_cyl(map, ainfo=ainfo)
	ainfo.lmul(alm, bl[:lmax+1], out=alm)
	return curvedsky.alm2map_cyl(alm, map, copy=True)

def get_distortion(shape, wcs):
	box = enmap.box(shape, wcs)
	dec1, dec2 = box[:,0]
	rmin = min(np.cos(dec1),np.cos(dec2))
	rmax = 1 if not dec1*dec2 > 0 else max(np.cos(dec1),np.cos(dec2))
	return rmax/rmin-1

def apply_beam(map, bl, max_distortion=0.1):
	if get_distortion(map.shape, map.wcs) > max_distortion:
		return apply_beam_sht(map, bl)
	else:
		return apply_beam_fft(map, bl)

def hget(fname):
	res = bunch.Bunch()
	with h5py.File(fname, "r") as hfile:
		for key in hfile:
			res[key] = hfile[key].value
	return res

def hput(fname, info):
	with h5py.File(fname, "w") as hfile:
		for key in info:
			hfile[key] = info[key]

def get_pixsizemap_cyl(shape, wcs):
	# ra step is constant for cylindrical projections
	dra  = np.abs(wcs.wcs.cdelt[0])*utils.degree
	# get the dec for all the pixel edges
	decs = enmap.pix2sky(shape, wcs, [np.arange(shape[-2]+1)-0.5,np.zeros(shape[-2]+1)])[0]
	sins = np.sin(decs)
	sizes= np.abs((sins[1:]-sins[:-1]))*dra
	# make it broadcast with full maps
	sizes = sizes[:,None]
	return sizes

def get_lbeam_exact(r, br, lmax, tol=1e-10):
	"""Given a 1d beam br(r), compute the 1d beam transform bl(l) up to lmax.
	Exact but slow."""
	imax = np.where(br>=br[0]*tol)[0][-1]
	return healpy.sphtfunc.beam2bl(br[:imax], r[:imax], lmax)

def get_lbeam_flat(r, br, shape, wcs):
	"""Given a 1d beam br(r), compute the 2d beam transform bl(ly,lx) for
	the l-space of the map with the given shape, wcs, assuming a flat sky"""
	cpix = np.array(shape[-2:])//2-1
	cpos = enmap.pix2sky(shape, wcs, cpix)
	rmap = enmap.shift(enmap.modrmap(shape, wcs, cpos), -cpix)
	bmap = enmap.ndmap(np.interp(rmap, r, br, right=0), wcs)
	return enmap.fft(bmap)

class RmatOld:
	def __init__(self, shape, wcs, beam_profile, rfact, lmax=20e3, pow=1):
		self.pixarea = get_pixsizemap_cyl(shape, wcs)
		self.bl      = healpy.sphtfunc.beam2bl(beam_profile[1]**pow, beam_profile[0], lmax)
		# We don't really need to factorize out the beam area like this
		self.barea   = self.bl[0]
		self.bl     /= self.barea
		self.rfact   = rfact
		self.pow     = pow
	def apply(self, map):
		rmap = apply_beam((self.rfact**self.pow * self.barea/self.pixarea) * map, self.bl)
		return rmap

class Rmat:
	# Rmat represents a linear operator that takes us from pixel delta function
	# with a given flux (mJy) to an observed source centered on that pixel, in uK.
	# This is a convolution of the input map by a scaled beam:
	#  omap[opix] = sum_dpix imap[opix-dpix]*ubeam[dpix]*flux_factor
	# Yes, there's no pixarea factor as there would be in a normal integral. This
	# is because we have defined our input map in terms of pixel delta functions.
	#
	# We want to do this with harmonic space convolutions.
	# 1. ifft(fft(delta)*fft(beam))*npix**0.5 = source profile with 1 max
	#    If normalization is turned off, then replace npix**0.5 with npix**-1
	# 2. To go from 1 mJy delta @150 GHz to source in uK, just multiply this
	#    by rfact, giving
	#    R = fract*ifft(fft(beam)*fft(imap)).real*npix**0.5
	# 3. But we don't work with 2d beams. We get a 1d real-space beam
	#    profile and turn it into a 1d l profile using beam2bl
	#    beam2bl returns a beam where b[0] is the beam area, while
	#    fft(beam) gives npix**0.5 * beam area / pix_area. So with beam2bl the formula is
	#    R = rfact*ifft(beam2bl(beam1d)*fft(imap)).real / pix_area
	# 4. What does squaring the real-space beam look like here? Need to disentangle
	#    the beam itself from the convolution factors. The given a real-space kernel
	#    kmap and some input map imap, the convolution function is
	#    convolve(kmap,imap) = ifft(fft(kmap)*fft(imap)).real*npix**0.5
	#    So rfact is part of the beam that should be squared, while npix isn't
	# 5. For the beam2bl case, the convolution is
	#    convolve(kmap1d,imap) = ifft(beam2bl(kmap1d)*fft(imap)).real / pix_area
	#    so again fract is the only factor that's part of the beam and needs squaring
	# 6. What I had is equivalent to this, but with the factors spread out between two
	#    different functions. So R is already correct.
	def __init__(self, shape, wcs, beam_profile, rfact, lmax=20e3, pow=1, max_distortion=0.1):
		self.pixarea = get_pixsizemap_cyl(shape, wcs)
		self.r       = beam_profile[0]
		self.rbeam   = (beam_profile[1]*rfact)**pow
		self.flat    = get_distortion(shape, wcs) < max_distortion
		# get_lbeam_exact uses beam2bl from healpix, which is very slow, so avoid it if possible
		if self.flat: self.lbeam = get_lbeam_flat (self.r, self.rbeam, shape, wcs)
		else:         self.lbeam = get_lbeam_exact(self.r, self.rbeam, lmax)
	def apply(self, map):
		if self.flat: return enmap.ifft(self.lbeam*enmap.fft(map)).real*map.npix**0.5
		else:         return apply_beam_sht(map, self.lbeam)/self.pixarea

def get_smooth_normalization(frhs, kmap, res=120, tol=2, bsize=1200):
	"""Split into cells of size res*res. For each cell,
	find the normalization factor needed to make the
	mean chisquare 1, where chisquare = frhs**2 / kmap.
	Gapfill outliers, and then interpolate back to full
	resolution. The result will be what kmap needs to be
	multiplied by to get the right chisquare.
	
	This could be made more robust by using a median of mean
	technique in each cell. That would avoid penalizing cells with
	local high signal. But as long as we are noise dominated the
	current approach should be good enough."""
	ny, nx = np.array(frhs.shape[-2:])//res
	mask   = kmap > np.max(kmap)*1e-4
	rblock = frhs[:ny*res,:nx*res].reshape(ny,res,nx,res)
	kblock = kmap[:ny*res,:nx*res].reshape(ny,res,nx,res)
	mblock = mask[:ny*res,:nx*res].reshape(ny,res,nx,res)
	# compute the mean chisquare per block
	with utils.nowarn():
		chisqs = rblock**2 / kblock
		chisqs[~mblock] = 0
		ngood         = np.sum(mblock,(1,3))
		mean_chisqs   = np.sum(chisqs,(1,3))/ngood
	# replace bad entries with typical value
	nmin = res**2 / 3
	good = ngood > nmin
	if np.all(~good): return enmap.full(frhs.shape, frhs.wcs, 1.0, frhs.dtype)
	ref   = np.median(mean_chisqs[good])
	with utils.nowarn():
		good &= np.abs(np.log(mean_chisqs/ref)) < tol
	if np.all(~good): mean_chisqs[:] = ref
	else: mean_chisqs[~good] = np.median(mean_chisqs[good])
	# Turn mean chisqs into kmap scaling factor to get the right chisquare
	norm_lowres = mean_chisqs
	norm_full   = enmap.zeros(frhs.shape, frhs.wcs, frhs.dtype)
	# This loop is just to save memory
	for y1 in range(0, frhs.shape[0], bsize):
		work = norm_full[y1:y1+bsize]
		opix = work.pixmap()
		opix[0] += y1
		ipix = opix/float(res)
		work[:] = norm_lowres.at(ipix, unit="pix", order=1, mode="nearest")
	return norm_full

def grow_mask(mask, n):
	return ndimage.distance_transform_edt(~mask) <= n

def solve(rhs, kmap, return_mask=False):
	with utils.nowarn():
		sigma = rhs / kmap**0.5
	mask = kmap < np.max(kmap)*1e-2
	sigma[mask] = 0
	return sigma if not return_mask else (sigma, mask)

#def get_mjd(idir):
#	with h5py.File(idir + "/info.hdf", "r") as hfile:
#		return hfile["mjd"].value
#
#def group_by_year(mjds, dur=365.24, nhist=12):
#	# First find the best splitting point via histogram
#	mjds   = np.asarray(mjds)
#	pix    = (mjds/dur*nhist).astype(int) % nhist
#	hist   = np.bincount(pix, minlength=nhist)
#	mjd_split = np.argmin(hist)*dur/nhist
#	# Then do the actual splitting
#	group_ind  = ((mjds-mjd_split)/dur).astype(int)
#	return utils.find_equal_groups(group_ind)
#
#def group_neighbors(mjds, tol=1):
#	return utils.find_equal_groups(mjds, tol)

def find_candidates(sigma, params, snmin=5, pad=0):
	sigma, params = unpad(sigma, pad), unpad(params, pad)
	try: snmin = unpad(snmin, pad)
	except: pass
	labels, nlabel = ndimage.label(sigma >= snmin)
	if nlabel < 1: return np.zeros([0,6])
	active = np.arange(1,nlabel+1)
	pixs   = np.array(ndimage.maximum_position(sigma, labels, active)).T
	sig    = sigma[pixs[0],pixs[1]]
	r, vy, vx, ivar = params[:,pixs[0],pixs[1]]
	damp   = ivar**-0.5
	amp    = sig*damp
	poss   = sigma.pix2sky(pixs)
	res    = np.array([poss[0],poss[1],sig,amp,damp,r,vy,vx]).T
	return res

def get_maxgauss_quantile(mean, nsigma):
	n =  statdist.maxgauss_n(mean)
	# Could have used +sigma and maxgauss_quant, but that would have more numerical issues
	p =  stats.norm.cdf(-nsigma)
	q = -statdist.mingauss_quant(p, n)
	return q

def fit_tail_gauss(data, fmin=0, fmax=3, vmin=-10, vmax=10, dv=0.05, rel_tol=0.1, abs_tol=1e-3, minhit=10,minsamp=0):
	default_params = [1,0,1]
	data = np.asarray(data).reshape(-1)
	nbin = int((vmax-vmin)/dv)
	pix  = np.maximum(0,np.minimum(nbin-1,((data-vmin)/dv).astype(int)))
	hist = np.bincount(pix, minlength=nbin).astype(float)
	# The edges can accumulate samples. Get rid of them
	hist[0] = hist[-1] = 0
	norm = np.max(hist)
	if norm == 0 or data.size < minsamp: return default_params
	hist /= norm
	# Extract the tail we will do the fit on
	maxpos  = np.argmax(hist)
	imin    = maxpos + utils.nint(fmin/dv)
	imax    = maxpos + utils.nint(fmax/dv)+1
	nhit = np.sum(hist[imin:imax]>0)
	if nhit < minhit: return default_params
	default_params[1] = vmin + maxpos*dv
	# Set up our errors. We have a term that's proportional to hist itself
	# both because higher errors there are expected due to poisson errors
	# (though these would go as sqrt), and because we want to avoid having
	# the top dominate the fit. We also have a fixed term to avoid having
	# empty bins at high values get infinite significance.
	dhist = hist*rel_tol + abs_tol
	x = (np.arange(nbin)+0.5)*dv+vmin
	def calc_chisq(params):
		A, mu, sigma = params
		model = A*np.exp(-0.5*((x-mu)/sigma)**2)
		chisq = np.sum(((hist-model)/dhist)**2)
		return chisq
	A, mu, sigma = optimize.fmin_powell(calc_chisq, default_params, disp=False)
	sigma = np.abs(sigma)
	return A, mu, sigma

def build_dist_map(sigma_max, mask=None, bsize=120, maskval=0):
	# Build a [mu,sigma] map approximating the statistical properties of the
	# nois across the map. Done by fitting a gaussian to the upper side of
	# the values in each tile.
	shape  = np.array(sigma_max.shape[-2:])
	if mask is None:
		if maskval is not None: mask = sigma_max != maskval
		else:                   mask = np.full(shape, True, bool)
	nblock = (shape+bsize-1)//bsize
	by, bx = (shape+nblock-1)//nblock
	# First estimate the statistics of each cell
	params = []
	for y in range(0, shape[0], by):
		for x in range(0, shape[1], bx):
			subs, subm = sigma_max[y:y+by,x:x+bx], mask[y:y+by,x:x+bx]
			A, mu, sigma = fit_tail_gauss(subs[subm], minsamp=1000)
			params.append([A,mu,sigma])
	# Replace bad fits with typical values
	params = np.array(params)
	bad    = (params[:,1] == 0)|(params[:,0]<0.5)
	ngood  = len(params)-np.sum(bad)
	if ngood == 0:
		return snmin_guass.copy()
	refpar = np.median(params[~bad,:],0)
	params[bad,:] = refpar
	omap = enmap.zeros((2,)+sigma_max.shape[-2:], sigma_max.wcs, sigma_max.dtype)
	# Then loop through and insert them into snmin
	i = 0
	for y in range(0, shape[0], by):
		for x in range(0, shape[1], bx):
			omap[:,y:y+by,x:x+bx] = params[i,1:,None,None]
			i += 1
	return omap

def unpad(map, pad): return map[...,pad:map.shape[-2]-pad,pad:map.shape[-1]-pad]

############################### old stuff - will be removed ################################

# This module contains functions used to implement the motion-compensated coordinate
# system used for the Planet 9 search. We use mjd throughout to be consistent with
# the coordinates module.

import numpy as np, ephem
from . import utils, parallax, ephemeris, coordinates, fft
from scipy import interpolate, special

def smooth(arr, n):
	arr = np.array(arr)
	fa  = fft.rfft(arr)
	fa[n:] = 0
	fft.irfft(fa, arr, normalize=True)
	return arr

def fourier_deriv(arr, order, dt=1):
	# Our fourier convention is
	# da/dx = d/dx sum exp(2 pi 1j x f/N) fa(f) = sum 2 pi 1j f/N exp(2 pi 1j x f/N) fa(f)
	# = ifft(2 pi 1j f/N fa)
	arr = np.array(arr)
	n   = arr.shape[-1]
	fa  = fft.rfft(arr)
	f   = fft.rfftfreq(n, dt)
	fa *= (2j*np.pi*f)**order
	fft.irfft(fa, arr, normalize=True)
	return arr

def orb_subsample(t, step=0.1, nsub=3):
	offs = (np.arange(nsub)-(nsub-1)/2.0)*step
	return (t[:,None] + offs[None,:]).reshape(-1)

# Notes:
#
# 1. The small earth parallax residual results in a small discontiunity
#    at the wraparound, which leads an extreme value for the speed at the
#    edge. This leads to spline problems. The discontinuity also leads to
#    ringing when smoothing. Could deal with this by padding beyond one orbit
# 2. It's really annoying that this should be so hard. Maybe it's easier to
#    just skip pyephem and compute the ideal orbit myself.
# 3. The main question we're asking here is:
#     if the object was at position p at time t1, where was it at time t0?
#    Is it possible to answer this question directly with pyephem, without
#    any manual parallax work?

class MotionCompensator:
	def __init__(self, obj=None, nsamp=1000, nderiv=2):
		if obj is None:
			obj = ephemeris.make_object(a=500, e=0.25, inc=20, Omega=90, omega=150)
		self.o    = obj
		self.tref = ephemeris.djd2mjd(obj._epoch)
		# Evaluate a coarse, smooth orbit.
		dt   = self.period / nsamp
		t    = self.tref + (np.arange(nsamp)-nsamp//2)*dt
		pos  = ephemeris.trace_orbit(obj, t, self.tref, nsub=20)
		# Find the normal vector for the orbit. We use a lot of cross products to make it
		# robust to deviations from ellipticity
		vs    = utils.ang2rect(pos[:2], zenith=False)
		vzen  = np.mean(np.cross(vs, np.roll(vs,-1,1),0,0),0)
		self.pzen = utils.rect2ang(vzen, zenith=False)
		# Transform the orbit to orbit-aligned coordinates so that the rotation is purely
		# in the lon direction
		pos_oo = utils.unwind(coordinates.recenter(pos[:2], self.pzen))
		# Now use the sub-points to compute the lon, dlon/dt and ddlon/dtdt
		lon     = pos_oo[0]
		sundist = pos[2,:]
		# Hack: smooth lon and sundist to avoid pyephem-induced jitter that
		# especially affects the speed and accel. We have to extract the average
		# orbital motion before smoothing, to avoid discontinuity
		nkeep     = 20
		avg_speed = 2*np.pi/self.period
		delta_lon = lon - avg_speed * np.arange(nsamp)*dt
		delta_lon = smooth(delta_lon, nkeep)
		#sundist   = smooth(sundist,   nkeep)
		lon       = delta_lon + avg_speed * np.arange(nsamp)*dt
		# Compute and spline the derivatives
		self.lon0   = lon[0]
		x           = utils.unwind(lon-self.lon0)
		self.dist_spline   = interpolate.splrep(x, sundist)
		self.deriv_splines = [interpolate.splrep(x, fourier_deriv(delta_lon, i+1, dt=dt) + (avg_speed if i == 0 else 0)) for i in range(nderiv)]
		# Debug
		self.lon = lon
		#self.speed = speed
		#self.accel = accel
		self.sundist = sundist
		self.pos_oo = pos_oo
		self.t = t
		self.dt = dt
		self.pos = pos
	@property
	def period(self):
		return ephemeris.yr * self.o._a**1.5
	def compensate(self, pos, t, tref, nit=2):
		"""Approximately compensate for how much an object currently
		at position pos has moved since the reference time tref, assuming
		it has a similar orbit to us. Compensates for both orbital motion
		and parallax."""
		# First find the sun distance and sun-relative coordinates
		pos_sunrel = pos
		for i in range(nit):
			pos_oo  = coordinates.recenter(pos_sunrel, self.pzen)
			x       = (pos_oo[0]-self.lon0)%(2*np.pi)
			sundist = interpolate.splev(x, self.dist_spline)
			pos_sunrel, earthdist = parallax.earth2sun_mixed(pos, sundist, t)
		# Then apply the orbital correction
		pos_oo  = coordinates.recenter(pos_sunrel, self.pzen)
		x       = (pos_oo[0]-self.lon0)%(2*np.pi)
		delta_t = t-tref
		old = pos_oo.copy()
		for i, spline in enumerate(self.deriv_splines):
			deriv = interpolate.splev(x, spline)
			pos_oo[0] -= delta_t**(i+1)/special.factorial(i+1) * deriv
		# Transform back to celestial coordinates
		opos = coordinates.decenter(pos_oo, self.pzen)
		return opos
