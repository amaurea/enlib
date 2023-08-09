from __future__ import division, print_function
import numpy as np, time, os, sys, healpy
from . import utils
with utils.nowarn(): import h5py
from scipy import ndimage, stats, spatial, integrate, optimize
from . import enmap, utils, curvedsky, bunch, parallax, cython, ephemeris, statdist, interpol
from . import nmat, pmat, sampcut, fft
from pixell import wcsutils

try: basestring
except: basestring = str

ym   = utils.arcmin/utils.yr2days

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

def p9_like(ra, dec, r=None, vy=None, vx=None, posonly=False, rmin=280, rmax=2000, emax=0.55,
		mininc=10*utils.degree, maxinc=30*utils.degree, dv=0.1*ym):
	"""Return whether a set of p9 parameters roughly match the p9
	hypothesis. Ideally we would do this test directly in terms of
	orbital elements, but for now we will just impose the simple
	criteria that the position shouldn't be further than the
	inclination from the ecliptic, and that the velocity should be
	roughly aligned with the ecliptic."""
	# Transform to ecliptic coordinates
	l,  b  = coordinates.transform("cel", "ecl", [ra,dec])
	if posonly: return np.abs(b) <= maxinc
	l2, b2 = coordinates.transform("cel", "ecl", [ra+vx/np.cos(dec),dec+vy])
	vl, vb = l2-l, b2-b
	good   = np.abs(b) <= maxinc
	# The object moves in a sine wave b(l) = inc*cos(l+phi). The velocity is
	# db/dt = A*sin(l+phi)*dl/dt => db/dl = inc*sin(l+phi)
	# We also see that b**2 + (db/dl)**2 = inc**2. This is a strictly more
	# powerful bound, so we don't need to restrict db/dl by itself.
	with utils.nowarn():
		vb_safe = np.maximum(0,np.abs(vb)-dv/2)
		vl_safe = np.maximum(0,np.abs(vl)-dv/2)
		inc = (b**2 + (vb_safe/vl_safe)**2)**0.5
		inc[np.isnan(inc)] = 0
		good  &= (inc >= mininc) & (inc <= maxinc)
	del inc
	# We can also place bounds on the distance and velocity
	good  &= (r >= rmin) & (r <= rmax)
	# An object with some semimajor axis a and eccentricity e will move
	# between r = a*(1-e) and r = a*(1+e). The angular velocity is
	# |v| = 360*60 * [(a/au)*(1-e**2)]**0.5 / (r/au)**2   arcmin/year
	# If we just know r, then the highest possible speed we could have
	# is if e and a are maximal and we are at perihelion: a = r/(1-emax),
	# and the lowest possible speed is if they are still maximal but we
	# are at aphelion: a = r/(1+emax)
	def calc_v(r,a,e): return 360*60 * (a*(1-e**2))**0.5 / r**2 * ym
	vmin  = calc_v(r, r/(1+emax), emax)
	vmax  = calc_v(r, r/(1-emax), emax)
	v2    = vy**2+vx**2
	good &= (v2 >= vmin**2) & (v2 <= vmax**2)
	del v2
	return good

#def cut_source_groups(srcs, rlim=2*utils.arcmin):
#	"""Handling very nearby point sources requires joint fitting, which is
#	a bit involved, and heavy. Such nearby groups usually occur either as
#	small residuals very close to very bright sources, or due to an extended
#	shape of a source or some other reason for a poor fit. 

def merge_nearby(srcs, rlim=2*utils.arcmin):
	"""given source parameters which might contain duplicates, detect these duplicates
	and merge them to produce a single catalog with no duplicates. sources are considered
	duplicates if they are within rlim of each other."""
	pos    = np.array([srcs[:,1]*np.cos(srcs[:,0]), srcs[:,0]]).T
	tree   = spatial.cKDTree(pos)
	groups = tree.query_ball_tree(tree, rlim)
	done   = np.zeros(len(srcs),bool)
	ocat   = []
	nmerged = []
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
		nmerged.append(len(group))
	ocat = np.array(ocat)
	nmerged = np.array(nmerged)
	return ocat, nmerged

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

def cut_bright_srcs_daytime(scan, srcs, alim_include=1e4, alim_size=10, errbox=[[-1,-2],[1,4]], step=0.5):
	"""Cut sources in srcs that are brighter than alim in uK. errbox is [[x1,y1],[x2,y2]] in arcminutes"""
	# Daytime sometimes has nasty multi-modal beams which we can't model
	# properly. To cut sources properly we need to cut the whole area that
	# could be contaminated by the beam. We do this by repeating the night-time
	# cut over a range pointing offsets corresponding the maximum extra size of
	# the beam. These offset ranges are similar to those of the actual pointing offsets:
	# About +- 1 arcmin in x and -1 to +3 arcmin in y.
	srcs_cut = srcs[srcs[:,2]>alim_include]
	if len(srcs) == 0: return scan
	tod  = np.zeros((scan.ndet,scan.nsamp), np.float32)
	maxerr = np.max(np.abs(errbox))
	# Since building PmatPtsrc is slow, we build it once and reuse it with
	# different pointing offsets by hackily modifying scan.offsets, which
	# it depends on. Not very elegant.
	psrc = pmat.PmatPtsrc(scan, srcs_cut, interpol_pad=maxerr, tmul=0)
	(x1,y1),(x2,y2) = np.array(errbox)
	el      = scan.boresight[0,2]
	offsets = scan.offsets.copy()
	raw_offs= coordinates.transform("tele","bore", (offsets[:,1:]+[0,el]).T, bore=[0,el,0,0]).T
	nx = utils.nint((x2-x1)/float(step))+1
	ny = utils.nint((y2-y1)/float(step))+1
	for x in np.linspace(x1,x2,nx)*utils.arcmin:
		for y in np.linspace(y1,y2,ny)*utils.arcmin:
			scan.offsets[:,1:] = coordinates.transform("bore","tele", (raw_offs-[x,y]).T, bore=[0,el,0,0]).T - [0,el]
			psrc.forward(tod, srcs)
			cut  = sampcut.from_mask(tod > alim_size)
			scan.cut *= cut
			# Ensure that our bright sources get gapfilled before anything else
			scan.d.cut_basic *= cut
		scan.offsets = offsets
		return scan

def make_tophat_beam(r, npoint=1000):
	"""Return a dummy beam that's 1 up to a radius of r, and then zero. Useful for
	simulating top hat shaped sources or cutting sources up to a given radius. The beam
	will be equispaced"""
	beam    = np.zeros([2,npoint])
	beam[0] = np.arange(npoint)*r/(npoint-2)
	beam[1,:-1] = 1
	beam[1,-1]  = 0
	return beam

def cut_srcs_rad(scan, srcs, r=4*utils.arcmin):
	"""Cut all sources in srcs out to a radius of r, regardless of their amplitude"""
	tod  = np.zeros((scan.ndet,scan.nsamp), np.float32)
	beam = make_tophat_beam(r)
	# Set the source amplitude to 1 in T, 0 in Q, U
	srcs_uniform = srcs.copy()
	srcs_uniform[:,2]   = 1
	srcs_uniform[:,3:5] = 0
	psrc = pmat.PmatPtsrc(scan, srcs_uniform, beam=beam)
	psrc.forward(tod, srcs_uniform)
	cut  = sampcut.from_mask(tod > 0.5)
	scan.cut *= cut
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
	alm  = curvedsky.map2alm_cyl(map, lmax=lmax)
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
			res[key] = hfile[key][()]
	return res

def hput(fname, info):
	with h5py.File(fname, "w") as hfile:
		for key in info:
			hfile[key] = info[key]

def get_pixsizemap_cyl(shape, wcs):
	# This function can be replaced by enmap.pixsizemap(shape, wcs, broadcastable=True)
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
	cpix   = np.array(shape[-2:])//2-1
	cpos   = enmap.pix2sky(shape, wcs, cpix)
	rmap   = enmap.shift(enmap.modrmap(shape, wcs, cpos), -cpix)
	bmap   = enmap.ndmap(np.interp(rmap, r, br, right=0), wcs)
	# Normalize the beam so that l=0 corresponds to the sky mean of the beam,
	# like it is for get_lbeam_exact
	lbeam  = enmap.fft(bmap, normalize=False).real
	lbeam *= lbeam.pixsize()
	return lbeam

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
	def __init__(self, shape, wcs, beam_profile, rfact, lmax=20e3, lknee1=0, alpha1=-5, lknee2=0, alpha2=-10,
			pow=1, max_distortion=0.1):
		beam_profile = np.array(beam_profile)
		beam_profile[1] = (beam_profile[1]*rfact)**pow
		self.beam    = Beam(shape, wcs, beam_profile, lmax=lmax, max_distortion=max_distortion)
		self.pixarea = get_pixsizemap_cyl(shape, wcs)
		# Allow extra filtering with a butterworth filter. Ideally this wouldn't be necessary, but
		# the map-maker noise model underestimates the low-l correlated noise significantly, so
		# we need this extra filtering to avoid leaking through too much atmospheric noise.
		# At the same time, estimate what fraction q of the remaining signal (after N" and B) this removes.
		#
		# Normally we have flux = mean(beam*rhs)/kmap having the right units.
		# Now we apply the extra filter Fatm, getting flux' = mean(beam*Fatm*rhs)/kmap
		# kmap cancels. rhs approx beam*Falready*mean_div. This gives us
		# flux = mean(beam**2*Falready*mean_div), flux' = mean(beam**2*Fatm*Falready*mean_div)
		# Hence flux'/flux approx mean(beam**2*Fatm*Falready)/mean(beam**2*Falready)
		# (since mean_div is just a number).
		if lknee1 > 0:
			l, lbeam, nmode = self.beam.l, self.beam.lbeam, self.beam.nmode
			Fatm     = butterworth(l, lknee1, alpha1)
			Falready = butterworth(l, lknee2, alpha2)**0.5
			self.q   = np.mean(Fatm*lbeam**2*Falready*nmode)/np.mean(lbeam**2*Falready*nmode)
			self.beam.lbeam *= Fatm
		else: self.q = 1.0
	def apply(self, map):
		# The pixarea thing takes us from mean-normalization to peak normalization
		return (self.beam.apply(map)/self.pixarea).astype(map.dtype)

# It would be nice to split up the FFT/SHT and beam functionality of this class
class Beam:
	def __init__(self, shape, wcs, beam_profile, flat="auto", lmax=20e3, max_distortion=0.1):
		self.r       = beam_profile[0]
		self.rbeam   = beam_profile[1]
		self.flat    = flat
		if self.flat == "auto":
			self.flat = get_distortion(shape, wcs) < max_distortion
		self.lmax    = utils.nint(lmax)
		# get_lbeam_exact uses beam2bl from healpix, which is very slow, so avoid it if possible
		if self.flat:
			self.lbeam = get_lbeam_flat (self.r, self.rbeam, shape, wcs)
			self.l     = self.lbeam.modlmap()
			self.nmode = 1
		else:
			self.lbeam = get_lbeam_exact(self.r, self.rbeam, self.lmax)
			self.l     = np.arange(self.lbeam.size)
			self.nmode = 2*self.l+1
	def apply(self, map, lfilter=None):
		"""If called as beam.apply(map), will return map convolved with the beam.
		What to convolve with can be overridden using the lfilter argument. The default
		corresponds to beam.apply(map, beam.lbeam). lfilter should correspond to the
		multipoles in beam.l, which can be either 2d or 1d depending on whether self.flat
		is True or False."""
		if lfilter is None: lfilter = self.lbeam
		if self.flat: return enmap.ifft(lfilter*enmap.fft(map)).real.astype(map.dtype)
		else:         return apply_beam_sht(map, lfilter).astype(map.dtype)

def butterworth(l, lknee, alpha):
	with utils.nowarn():
		return 1/(1 + (l/lknee)**alpha)

def get_rough_powspec(map, mask, tsize=240, ntile=32, hit_tol=0.5):
	"""Estimate a quick and dirty power spectrum from map. This should be fast no matter
	how big the map is. It works by selecting only a limited number of randomly chosen
	tiles from a map, and returns their mean spectrum"""
	dl     = utils.nint(360/np.min(np.abs(map.wcs.wcs.cdelt))/tsize)
	ny, nx = np.array(map.shape[-2:])//tsize
	inds   = [(ty,tx) for ty in range(ny) for tx in range(nx)]
	np.random.shuffle(inds)
	specs, ls = [], []
	nbin   = np.inf
	for ty, tx in inds:
		y1, y2 = ty*tsize, (ty+1)*tsize
		x1, x2 = tx*tsize, (tx+1)*tsize
		submask = mask[y1:y2,x1:x2]
		submap  = map[...,y1:y2,x1:x2]
		if np.mean(submask) < hit_tol: continue
		ps2d    = np.abs(enmap.fft(submap))**2
		spec, l = ps2d.lbin(bsize=dl)
		specs.append(spec)
		ls.append(l)
		nbin = min(nbin, l.size)
		if len(specs) > ntile: break
	if len(specs) == 0:
		raise ValueError("Can't estimate spectrum of overly masked map")
	specs = np.mean([spec[:nbin] for spec in specs],0)
	ls    = ls[0][:nbin]
	return specs, ls

def estimate_lknee(ps, l, lknee0=1000, alpha=-4):
	"""Estimate lknee from the power spectrum ps, which is sampled at the given ls"""
	# We might have a falling part at the left due to convergence etc. So restrict the
	# fit to starting from the peak
	i1 = np.argmax(ps)
	ps2, l2 = ps[i1:], l[i1:]
	# We will do the fit in log space
	ps2 = np.log(ps2)
	def calc_chisq(lknee):
		if lknee <= 0: return np.inf
		template = np.log(1+(np.maximum(l2,1)/lknee)**alpha)
		# Solve for the white noise level
		w = np.mean(ps2-template)
		model = template+w
		# Now that we have the model we can compute the chisquare
		chisq = np.sum((ps2-model)**2)
		#print("%8.2f %15.7e %15.7e" % (lknee, w, chisq))
		return chisq
	lknee = optimize.fmin_powell(calc_chisq, lknee0, disp=False)
	return lknee

def fit_double_butterworth(ps, l, lknee1=4000, alpha1=-5, lknee2=2000, alpha2=-10, lmin=1, verbose=False):
	"""Fit the function A/butter(l,l1,a1)*butter(l,l2,a2) to ps, where butter(l,lknee,alpha)
	= (1+(l/lknee)**alpha)**-1. Here the first butter could represent the rise of atmospheric
	noise, while the second one could represent the map-maker transfer function from stopping
	CG early."""
	# Cut out any too small values of l in the fit. These will lead to division by zero
	ps, l = ps[l>=lmin], l[l>=lmin]
	# Let the weight include a part that depends on the values themselves, to make the fit
	# more robust to large magnitude differences.
	weight = 1/(ps + np.median(ps))**2
	def butter(l,lknee,alpha): return 1/(1+(l/lknee)**alpha)
	def calc_chisq(x):
		l1, a1, l2, a2 = x
		# Would have returned np.inf, but that makes it print a noisy warning
		if l1 <= 0 or l2 <= 0 or a1 < -50 or a2 < -50 or a1 > 0 or a2 > 0: return 1e200
		template = butter(l,l2,a2)/butter(l,l1,a1)
		# Solve for the white noise level
		A     = np.sum(ps*weight*template)/np.sum(template**2*weight)
		model = A*template
		chisq = np.sum((ps-model)**2*weight)
		if verbose:
			print("%8.2f %8.3f %8.2f %8.3f %15.7e %15.7e" % (l1, a1, l2, a2, A, chisq))
		return chisq
	l1, a1, l2, a2 = optimize.fmin_powell(calc_chisq, [lknee1,alpha1,lknee2,alpha2], disp=False)
	template = butter(l,l2,a2)/butter(l,l1,a1)
	A        = np.sum(ps*weight*template)/np.sum(template**2*weight)
	return bunch.Bunch(lknee1=l1, alpha1=a1, lknee2=l2, alpha2=a2, A=A)

def setup_noise_fit(rhs, args_lknee=None, dirpath=None, disable=False, dump=False):
	"""Helper funciton. Here to avoid code repetition in tenki/planet9.py, not because
	it's a useful function in general"""
	if disable:
		return bunch.Bunch(lknee1=0, alpha1=-5, lknee2=0, alpha2=-10, A=1)
	if args_lknee is not None:
		try:
			lknee = float(args_lknee)
			return bunch.Bunch(lknee1=lknee, alpha1=-5, lknee2=lknee/2, alpha2=-10, A=1)
		except ValueError:
			rhs    = enmap.read_map(args_lknee + "/" + dirpath.split("/")[-1] + "/rhs.fits")
	try:
		ps, ls = get_rough_powspec(rhs, rhs != 0)
		fit    = fit_double_butterworth(ps, ls)
		if dump:
			model  = fit.A*butterworth(ls, fit.lknee2, fit.alpha2)/butterworth(ls+1e-9, fit.lknee1, fit.alpha1)
			np.savetxt(dirpath + "/test_spec.txt", np.array([ls,ps,model]).T, fmt="%15.7e",
					header="l1 %8.2f a1 %8.3f l2 %8.2f a2 %8.3f A %15.7e" % (
						fit.lknee1, fit.alpha1, fit.lknee2, fit.alpha2, fit.A))
		return fit
	except ValueError:
		return bunch.Bunch(lknee1=3000, alpha1=-5, lknee2=1500, alpha2=-10, A=1)

def get_normalization(frhs, kmap, res=240, hitlim=0.5):
	"""Compute the factor kmap must be multiplied with such that
	sigma = frhs/kmap**0.5 has a standard deviation of 1. Unlike
	get_smooth_normalization we only return a single number, not
	a position-dependent normalization.

	We do this by splitting the map into regions, cutting regions
	that aren't sufficiently hit, computing the standard deviation
	in each of the rest and using the median of these to compute
	the result. The median is used to make us more robust to
	outliers due to e.g. signal-dominated regions."""
	ny, nx = np.array(frhs.shape[-2:])//res
	mask   = kmap > np.max(kmap)*1e-4
	r2block = frhs[:ny*res,:nx*res].reshape(ny,res,nx,res)**2
	kblock  = kmap[:ny*res,:nx*res].reshape(ny,res,nx,res)
	mblock  = mask[:ny*res,:nx*res].reshape(ny,res,nx,res)
	# We want to solve for the the factor a such that r2block = a*kblock.
	# This is a = sum(kblock*r2block)/sum(kblock**2)
	with utils.nowarn():
		avals = np.sum(kblock*r2block,(1,3))/np.sum(kblock**2,(1,3))
	hitfrac = np.mean(mblock,(1,3))
	good    = hitfrac >= hitlim
	if np.sum(good) > 0:
		# Get the median of the acceptable avals
		avals   = avals[hitfrac >= hitlim]
		a       = np.median(avals)
	else:
		# Hm, nothing was hit enough. If so, just do a single overall mean
		a = np.sum(kblock*r2block)/np.sum(kblock**2)
	return a

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
	"""Grow positive-bad mask by n pixels"""
	return enmap.samewcs(ndimage.distance_transform_edt(~mask) <= n, mask)

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

def find_candidates(sigma, params, hitmap=None, snmin=5, pad=0):
	sigma, params = unpad(sigma, pad), unpad(params, pad)
	if hitmap is not None: hitmap = unpad(hitmap, pad)
	try: snmin = unpad(snmin, pad)
	except: pass
	labels, nlabel = ndimage.label(sigma >= snmin)
	if nlabel < 1: return np.zeros([0,11])
	active = np.arange(1,nlabel+1)
	pixs   = np.array(ndimage.maximum_position(sigma, labels, active)).T
	sig    = sigma[pixs[0],pixs[1]]
	r, vy, vx, rhs, ivar = params[:,pixs[0],pixs[1]]
	if hitmap is not None:
		hits   = hitmap[pixs[0],pixs[1]]
	else:
		hits   = r*0
	damp   = ivar**-0.5
	amp    = rhs/ivar
	poss   = sigma.pix2sky(pixs)
	res    = np.array([poss[0],poss[1],sig,amp,damp,r,vy,vx,hits,rhs,ivar]).T
	return res

def build_mu_sigma_map(imap, mask=None, bsize=(30,240), maskval=0, default=(2, 1)):
	"""Estimate the mean (mu) and standard deviation (sigma) as a function of position
	of the input map (imap). Returns mu, sigma. These can be used to normalize the
	input map as such: (imap-mu)/sigma.
	
	This is the first step in normalizing the
	planet 9 search detection distribution, but it is not enough by itself due to
	the non-gaussianity of that distribution."""
	shape  = np.array(imap.shape[-2:])
	bsize  = np.zeros(2,int)+bsize
	if mask is None:
		if maskval is not None: mask = imap != maskval
		else:                   mask = np.full(shape, True, bool)
	nblock = (shape+bsize-1)//bsize
	by, bx = (shape+nblock-1)//nblock
	# First estimate the statistics of each cell
	params = []
	for y in range(0, shape[0], by):
		for x in range(0, shape[1], bx):
			subs, subm = imap[y:y+by,x:x+bx], mask[y:y+by,x:x+bx]
			vals = subs[subm]
			if vals.size < 4000: params.append([0,1])
			else: params.append([np.mean(vals), np.std(vals)])
	omap = enmap.zeros((2,)+imap.shape[-2:], imap.wcs, imap.dtype)
	# Replace bad fits with typical values
	params = np.array(params)
	bad    = params[:,0] == 0
	ngood  = len(params)-np.sum(bad)
	if ngood == 0:
		# use default value if we couldn't measure anything at all.
		# A mean of 2 and dev of 1/3 is typical. We use (2,1) here to be conservative.
		omap[0] = default[0]
		omap[1] = default[1]
	else:
		refpar = np.median(params[~bad,:],0)
		params[bad,:] = refpar
		# Then loop through and insert them into omap
		i = 0
		for y in range(0, shape[0], by):
			for x in range(0, shape[1], bx):
				omap[:,y:y+by,x:x+bx] = params[i,:,None,None]
				i += 1
	return omap # [{mu,sigma},ny,nx]

def find_sf_correction(snmap, mask=None, ref_res=2*utils.arcmin, sncut=1, pmax_fit=0.1, nmin=100, icut=50):
	"""Given a S/N map, such as the one one gets by applying build_mu_sigma_map to a raw S/N map,
	find the transformation sn -> (sn-mu)/sigma one must apply such that the high-SN tail of the
	CDF matches that of a normal distribution."""
	from scipy.special import erf, erfinv
	if mask is None: mask = snmap != 0
	else: snmap = snmap * mask
	# Here's how the fit will work:
	# 1. find the peaks
	labels, n = ndimage.label(snmap > sncut)
	sn  = ndimage.maximum(snmap, labels, np.arange(n)+1)
	# 2. build their empirical distribution function
	sn  = np.sort(sn)[::-1]
	sf  = np.arange(n)+1
	# 3. Given the area we cover we can figure out what sn *should* be
	# to match the sf we actually observe for that value. This gives us
	# a correction: sn_corr = inv_sf_theory(sf_empirical(sn)).
	# sf_theory = ndof * 0.5*(1+erf(-x/2**0.5)), so
	# inv_sf_theory = -inv_erf(x*2/ndof-1)*2**0.5
	area = np.sum(mask.pixsizemap()*mask)
	ndof = utils.nint(area/ref_res**2)
	def find(a, v): return np.searchsorted(a, v)
	def inv_sf_theory(x): return -2**0.5 * erfinv(x*2/ndof-1)
	imin = min(utils.nint(n*pmax_fit), ndof) # skip values too close to peak or too big for inverse
	if imin-icut < nmin: return 0, 1
	sn, sf = sn[icut:imin], sf[icut:imin]
	targ_sn = inv_sf_theory(sf)
	# 4. Fit a linear model to sn: sn = targ_sn*sigma+mu
	P   = np.array([targ_sn,targ_sn*0+1])
	rhs = P.dot(sn)
	div = P.dot(P.T)
	mu, sigma = np.linalg.solve(div, rhs)[::-1]
	# Ok, we're done. Using this we can go from the observed sn to a more gaussian sn
	# via sn -> (sn-mu)/sigma.
	return mu, sigma

def unpad(map, pad): return map[...,pad:map.shape[-2]-pad,pad:map.shape[-1]-pad]

class SplineEphem:
	def __init__(self, mjd, ra, dec, r, name=None):
		self.name = name
		self.data = np.array([mjd,ra,dec,r],dtype=np.float)
		# Check that the mjds are equi-spaced
		dmjds = self.data[0,1:]-self.data[0,:-1]
		self.mjd1, self.mjd2, self.dmjd = self.data[0,0], self.data[0,-1], dmjds[0]
		assert np.all(np.abs(dmjds-self.dmjd)<1e-5), "mjd must be equi-spaced in SplineEphem"
		# Build spline for each
		self.spline = self.data[1:].copy()
		interpol.spline_filter(self.spline, border="mirror", ndim=1)
	@property
	def nsamp(self): return self.spline.shape[-1]
	def __call__(self, mjds):
		mjds = np.asarray(mjds)
		mjd1, mjd2 = utils.minmax(mjds)
		assert mjd1 >= self.mjd1, "mjd %f is outside the validity range %f to %f" % (mjd1, self.mjd1, self.mjd2)
		assert mjd2 <= self.mjd2, "mjd %f is outside the validity range %f to %f" % (mjd2, self.mjd1, self.mjd2)
		pix  = (mjds-self.mjd1)/self.dmjd
		return interpol.map_coordinates(self.spline, pix[None], border="mirror", prefilter=False)

def get_asteroids(fname, names=None):
	if fname is None: return None
	asteroid_set = read_asteroids(fname)
	if names is None: names = asteroid_set.keys()
	elif isinstance(names, basestring): names = names.split(",")
	asteroids = [asteroid_set[key] for key in names]
	return asteroids

def read_asteroids(fname):
	res = bunch.Bunch()
	with h5py.File(fname, "r") as hfile:
		for key in hfile:
			data = hfile[key][:].view(np.recarray)
			res[key] = SplineEphem(data.mjd, data.ra*utils.degree, data.dec*utils.degree, data.r, name=key)
	return res

def build_asteroid_mask(shape, wcs, asteroids, mjds, r=3*utils.arcmin):
	"""Build a mask with the geometry shape, wcs for the given list of EphemSplines "asteroids",
	masking all pixels hit by any of the asteroids for the given set of mjds, up to a radius
	of r"""
	mask = enmap.zeros(shape, wcs, bool)
	mjds = np.asarray(mjds)
	for ai, ast in enumerate(asteroids):
		a = ast(mjds)
		poss   = ast(mjds)[1::-1].T
		pboxes = enmap.neighborhood_pixboxes(shape, wcs, poss, r)
		for i, pbox in enumerate(pboxes):
			submap = mask.extract_pixbox(pbox)
			submap = submap.modrmap(poss[i])<=r
			mask.insert_at(pbox, submap, op=np.ndarray.__ior__)
	return mask

def cut_asteroids_scan(scan, asteroids, r=3*utils.arcmin):
	"""Cut the the samples that come within a distance of r in radians
	from the given asteroids. scan is modified in-place"""
	from enact import cuts as actcuts
	import time
	for ai, asteroid in enumerate(asteroids):
		apos  = asteroid(scan.mjd0)[:2]
		bore  = scan.boresight.T.copy()
		bore[0] = utils.mjd2ctime(scan.mjd0)+bore[0]
		t1 = time.time()
		scan.cut *= actcuts.avoidance_cut(bore, scan.offsets[:,1:], scan.site, apos, r)
		t2 = time.time()
	return scan

def build_planet_mask(shape, wcs, planets, mjds, r=50*utils.arcmin):
	"""Build a mask with the geometry shape, wcs for the given list of planet names,
	masking all pixels hit by any of the planets for the given set of mjds, up to a radius
	of r"""
	mask = enmap.zeros(shape, wcs, bool)
	mjds = np.asarray(mjds)
	for pi, pname in enumerate(planets):
		# Get the coordinates for each mjd
		poss   = ephemeris.ephem_raw(pname, mjds)[1::-1].T
		pboxes = enmap.neighborhood_pixboxes(shape, wcs, poss, r)
		for i, pbox in enumerate(pboxes):
			submap = mask.extract_pixbox(pbox)
			submap = submap.modrmap(poss[i])<=r
			mask.insert_at(pbox, submap, op=np.ndarray.__ior__)
	return mask

def overlaps(pbox1, pbox2, nphi=0):
	return len(utils.sbox_intersect(np.array(pbox1).T,np.array(pbox2).T,wrap=[0,nphi])) > 0

# Get the pixel bounding box of each input map in terms of our output area
def read_pboxes(idirs, wcs, inames=None, comm=None, verbose=False):
	if comm is None:
		from . import mpi
		comm = mpi.COMM_SELF
	pboxes = np.zeros((len(idirs),2,2),int)
	for ind in range(comm.rank, len(idirs), comm.size):
		name = inames[ind] if inames is not None else idirs[ind]
		if verbose: print("%3d Reading geometry %s" % (comm.rank, name))
		tshape, twcs = enmap.read_map_geometry(idirs[ind] + "/frhs.fits")
		pboxes[ind] = enmap.pixbox_of(wcs, tshape, twcs)
	pboxes = utils.allreduce(pboxes, comm)
	return pboxes

def get_geometry_file(area): return enmap.read_map_geometry(area)
def get_geometry_str(area):
	# dec1:dec2,ra1:ra2[,down] -> [[dec1,ra1],[dec2,ra2]]
	toks = area.split(",")
	down = int(toks[2]) if len(toks) > 2 else 1
	box = np.array([[float(w) for w in tok.split(":")] for tok in toks[:2]]).T*utils.degree
	box[:,1] = np.sort(box[:,1])[::-1] # standard ra ordering
	geo      = enmap.geometry(box, res=0.5*utils.arcmin, proj="car", ref=[0,0])
	geo_ref  = enmap.fullsky_geometry(res=0.5*utils.arcmin)
	if down > 1: geo = downgrade_geometry_compatible(*geo, geo_ref[1], down)
	return geo
def get_geometry(area):
	try: return get_geometry_str(area)
	except ValueError: return get_geometry_file(area)

def downgrade_compatible(map, ref_wcs, factor):
	"""Downgrade map by factor, but crop as necessary to make sure that
	we stay compatible with the given reference geometry if it's downgraded by the
	same factor"""
	orig_wcs = wcsutils.scale(ref_wcs, factor)
	y1, x1   = utils.nint(enmap.sky2pix(None, orig_wcs, map.pix2sky([0,0]), safe=False))
	omap     = enmap.downgrade(map[...,(-y1)%factor:,(-x1)%factor:], factor)
	return omap

def downgrade_geometry_compatible(shape, wcs, ref_wcs, factor):
	"""Downgrade geometry by factor, but crop as necessary to make sure that
	we stay compatible with the given reference geometry if it's downgraded by the
	same factor"""
	# Get the coordinates of our top-left corner in the reference geometry
	orig_wcs = wcsutils.scale(ref_wcs, factor)
	y1, x1 = utils.nint(enmap.sky2pix(None, orig_wcs, enmap.pix2sky(shape, wcs, [0,0]), safe=False))
	# These must be a multiple of factor for us to stay compatible, so slice as necessary
	shape, wcs = enmap.slice_geometry(shape, wcs, (slice((-y1)%factor,None),slice((-x1)%factor,None)))
	shape, wcs = enmap.downgrade_geometry(shape, wcs, factor)
	return shape, wcs

def calc_rlim(fluxlim, dists, fref, rref):
	"""Compute the distance limit for a given *distance-depencent* flux-limit
	fluxlim[nr,ny,nx], with the first axis corresponding to the distances dists[nr]"""
	dists = np.asarray(dists)
	# For a single flim, rlim = rref * (flim/fref)**-0.5, but things get more complicated
	# for an r-dependent flim. We start by computing the limit for each of them.
	ndist = len(dists)
	with utils.nowarn():
		rlims = rref * (fluxlim/fref)**-0.5
		rlims[~np.isfinite(rlims)] = 0
	# This gives us the rlim at each distance sample point. We can now loop through
	# each distance pair for which we can linearly interpolate:
	# rlim = rlim1 + (rlim2-rlim1)*x
	# r    = r1 + (r2-r1)*x
	# If the r and rlim curves don't cross, then this bin doesn't apply for the distance in
	# question. If they do cross, then that r = rlim is the one we want. In theory this
	# could happen in multiple bins, but it's pretty unlikely to.
	# there will be a crossing if (rlim1-r1)*(rlim2-r2) <= 0, and the crossing will be
	# at x = -(rlim1-r1)/((rlim2-r2)-(rlim1-r1)) = -roff1/(roff2-roff1)
	roff = rlims - dists[:,None,None]
	rlim = rlims[0]*0
	for i in range(ndist-1):
		roff1, roff2 = roff[i:i+2]
		crosses = roff1*roff2 <= 0
		x = -roff1/(roff2-roff1)
		rlim[crosses] = dists[i] + x[crosses]*(dists[i+1]-dists[i])
	del crosses
	# Handle edges
	inner = np.all(roff < 0, 0)
	rlim[inner] = rlims[0,inner]
	del inner
	outer = np.all(roff > 0, 0)
	rlim[outer] = rlims[-1,outer]
	return rlim

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
