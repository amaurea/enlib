"""This module contains implementations for "pointing matrices", which project
between time-ordered data (described by a Scan class) down to some other space,
such as a map. Its general interface is .forward(tod,m) and .backward(tod,m).
Both of these are allowed to modify both arguments, though usually
forward will update tod based on m, and backward will update m based on tod.
The reason for allowing the other argument to be modified is to make it easier
to incrementally project different parts of the signal.
"""
import numpy as np, time
from enlib import enmap, interpol, utils, coordinates, config, errors, array_ops
from enlib import parallax, bunch
import pmat_core_32
import pmat_core_64
def get_core(dtype):
	if dtype == np.float32:
		return pmat_core_32.pmat_core
	else:
		return pmat_core_64.pmat_core

config.default("pmat_map_order",      0, "The interpolation order of the map pointing matrix.")
config.default("pmat_cut_type",  "full", "The cut sample representation used. 'full' uses one degree of freedom for each cut sample. 'bin:N' uses one degree of freedom for every N samples. 'exp' used one degree of freedom for the first sample, then one for the next two, one for the next 4, and so on, giving high resoultion at the edges of each cut range, and low resolution in the middle.")
config.default("map_sys",       "equ", "The coordinate system of the maps. Can be eg. 'hor', 'equ' or 'gal'.")
config.default("pmat_accuracy",     1.0, "Factor by which to lower accuracy requirement in pointing interpolation. 1.0 corresponds to 1e-3 pixels and 0.1 arc minute in polangle")
config.default("pmat_interpol_max_size", 100000, "Maximum mesh size in pointing interpolation. Worst-case time and memory scale at most proportionally with this.")
config.default("pmat_interpol_max_time", 50, "Maximum time to spend in pointing interpolation constructor. Actual time spent may be up to twice this.")
# Disable window for now - it is buggy
config.default("tod_window",        0.0, "Seconds by which to window each end of the TOD.")

class PointingMatrix:
	def forward(self, tod, m): raise NotImplementedError
	def backward(self, tod, m): raise NotImplementedError

class PmatMap(PointingMatrix):
	"""Fortran-accelerated scan <-> enmap pointing matrix implementation.
	20 times faster than the slower python+numpy implementation below."""
	def __init__(self, scan, template, sys=None, order=None):
		sys        = config.get("map_sys", sys)
		transform  = pos2pix(scan,template,sys)
		ipol, obox = build_interpol(transform, scan.box, id=scan.entry.id)
		self.rbox, self.nbox, self.yvals = extract_interpol_params(ipol, template.dtype)
		# Use obox to extract a pixel bounding box for this scan.
		# These are the only pixels pmat needs to concern itself with.
		# Reducing the number of pixels makes us more memory efficient
		self.pixbox, self.nphi  = build_pixbox(obox[:,:2], template)
		self.scan,   self.dtype = scan, template.dtype
		self.core  = get_core(self.dtype)
		self.order = config.get("pmat_map_order", order)
	def forward(self, tod, m, tmul=1, mmul=1, times=None):
		"""m -> tod"""
		if times is None: times = np.zeros(5)
		self.core.pmat_map( 1, 1, self.order+1, tmul, mmul, tod.T, m.T, self.scan.boresight.T, self.scan.hwp_phase.T, self.scan.offsets.T, self.scan.comps.T, self.rbox.T, self.nbox, self.yvals.T, self.pixbox.T, self.nphi, times)
	def backward(self, tod, m, tmul=1, mmul=1, times=None):
		"""tod -> m"""
		if times is None: times = np.zeros(5)
		self.core.pmat_map(-1, 1, self.order+1, tmul, mmul, tod.T, m.T, self.scan.boresight.T, self.scan.hwp_phase.T, self.scan.offsets.T, self.scan.comps.T, self.rbox.T, self.nbox, self.yvals.T, self.pixbox.T, self.nphi, times)
	def translate(self, bore=None, offs=None, comps=None):
		"""Perform the coordinate transformation used in the pointing matrix without
		actually projecting TOD values to a map."""
		if bore  is None: bore  = self.scan.boresight
		if offs  is None: offs  = self.scan.offsets[:1]*0
		if comps is None: comps = self.scan.comps[:self.scan.offsets.shape[0]]*0
		bore, offs, comps = np.asarray(bore), np.asarray(offs), np.asarray(comps)
		nsamp, ndet, ncomp = bore.shape[0], offs.shape[0], comps.shape[1]
		dtype = self.dtype
		pix   = np.empty([ndet,nsamp,2],dtype=dtype)
		phase = np.empty([ndet,nsamp,ncomp],dtype=dtype)
		self.core.translate(bore.T, pix.T, phase.T, offs.T, comps.T, self.comps, self.rbox.T, self.nbox, self.yvals.T)
		return pix, phase

def get_scan_dir(az, step=3):
	"""The scanning direction is 1 if az is increasing, and 0 otherwise.
	Larger values of step are less sensitive to jitter, but have worse
	resolution."""
	sdir = az[step:]>az[:-step]
	return np.concatenate([
			np.full(step/2,sdir[0],dtype=bool),
			sdir,
			np.full(step-step/2,sdir[-1:],dtype=bool)
		])

def get_scan_period(az, srate=1):
	"""Estimate the period of the az sweeps. This is done by
	finding the interval between zero crossings of az-mean(az)."""
	off  = az - np.mean(az)
	up_cross = (off[1:]*off[:-1] < 0)&(off[1:]-off[:-1]>0)
	up_cross = np.where(up_cross)[0]
	periods  = up_cross[1:]-up_cross[:-1]
	if len(periods) == 0: return 0.0
	# Jitter may cause very short ones blips just as we are crossing.
	# Discard these.
	periods = periods[periods>np.mean(periods)*0.2]
	# And use the median of the reminder as our estimate
	period = np.median(periods) / float(srate)
	return period

def build_work_shift(transform, hor_box, scan_period):
	"""Given a transofrmation that takes [{t,az,el},nsamp] into [{y,x,...},nsamp],
	and a bounding box [{from,to},{t,az,el}], compute the parameters for a
	per-y shift in x that makes scans roughly straight. These parameters are returned
	in the form of wbox[{from,to},{y,x'}] and wshift[{up,down},ny]. They are used in the shifted
	pmat implementation."""
	# The problem with using the real scanning profile is that it can't be adjusted to
	# cover all the detectors, and at any y, the az where a detector hits that y will
	# be different. So only one of them can faithfully follow the profile after all.
	# So I think I'll stay with the simple model I use here, and just take the travel
	# time into account, through an extra parameters.

	# Find the pixel bounds corresponding to our hor bounds
	hor_corn = utils.box2corners(hor_box)
	pix_corn = transform(hor_corn.T)[:2].T
	pix_box  = utils.bounding_box(pix_corn)
	# The y bounds are the most relevant. Our wshift must
	# be defined for every output y in the range.
	y0 = int(np.floor(pix_box[0,0]))
	y1 = int(np.ceil(pix_box[1,0]))+1
	mean_t, mean_az, mean_el = np.mean(hor_box,0)
	# Get a forward and backwards sweep
	wshift = np.array([
		measure_sweep_pixels(transform, [mean_t,mean_t+scan_period], hor_box[:,1],    mean_el, [y0,y1]),
		measure_sweep_pixels(transform, [mean_t,mean_t+scan_period], hor_box[::-1,1], mean_el, [y0,y1])])
	# For each of these, find the pixel bounds. The total
	# bounds will be the union of these
	wboxes = []
	for wshift_single in wshift:
		# Use this shift to transform the pixel corners into shifted
		# coordinates. This wil give us the bounds of the shifted system
		shift_corn = pix_corn.copy()
		np.set_printoptions(suppress=True)
		print "pix_box"
		print pix_box
		print "y0", y0, "y1", y1
		print "shift_corn before"
		print shift_corn
		print "shift_box before"
		print utils.bounding_box(shift_corn)
		print "shifts args"
		print np.round(shift_corn[:,0]).astype(int)-y0
		print "shifts"
		print wshift_single[np.round(shift_corn[:,0]).astype(int)-y0]
		shift_corn[:,1] -= wshift_single[np.round(shift_corn[:,0]-y0).astype(int)]
		print "shift_corn after"
		print shift_corn
		print "shift_box after"
		print utils.bounding_box(shift_corn)
		wboxes.append(utils.bounding_box(shift_corn))
	# Merge wboxes
	wbox = utils.bounding_box(wboxes)
	wbox[0] = np.floor(wbox[0])
	wbox[1] = np.ceil (wbox[1])
	wbox    = wbox.astype(int)
	return wbox, wshift

import sys
def measure_sweep_pixels(transform, trange, azrange, el, yrange, padstep=None, nsamp=None, ntry=None):
	"""Helper function for build_work_shift. Measure the x for each y of an azimuth sweep."""
	if nsamp   is None: nsamp   = 10000
	if padstep is None: padstep = 4*utils.degree
	if ntry    is None: ntry    = 5
	y0, y1 = yrange
	pad = padstep
	for i in range(ntry):
		az0, az1 = utils.widen_box(azrange, pad, relative=False)
		ipos = np.zeros([3,nsamp])
		# Forward sweep
		ipos[0] = np.linspace(trange[0], trange[1], nsamp)
		ipos[1] = np.linspace(az0,       az1,       nsamp)
		ipos[2] = el
		opix = np.round(transform(ipos)[:2]).astype(int)
		#print "opix"
		#sys.stdout.flush()
		#np.savetxt("/dev/stdout", opix.T, "%6d")
		#sys.stdout.flush()
		# Get the entries with unique y values
		uy, ui = np.unique(opix[0], return_index=True)
		if uy[0] > y0 or uy[-1] <= y1:
			# We didn't cover the range, so try again
			pad += padstep
			continue
		# Restrict to relevant pixel range
		yi = ui[(uy>=y0)&(uy<y1)]
		if len(ui) < y1-y0:
			# We didn't hit every pixel. Try again
			nsamp *= 2
			continue
		wshift  = opix[1,yi]
		# Shift is defined to start at zero, since we will put
		# the absolute offset into wbox.
		wshift -= wshift[0]
		break
	else:
		# We didn't find a match! Just use a constant shift of zero.
		# This shouldn't happen, though.
		wshift = np.zeros(y1-y0,int)
	return wshift

class PmatMapMultibeam(PointingMatrix):
	"""Like PmatMap, but with multiple, displaced beams."""
	def __init__(self, scan, template, beam_offs, beam_comps, sys=None, order=None):
		# beam_offs has format [nbeam,ndet,{dt,dra,ddec,}], which allows
		# each detector to have a separate beam. The dt part is pretty useless.
		# beam_comps has format [nbeam,ndet,{T,Q,U}].
		# Get the full box after taking all beam offsets into account
		ibox = np.array([np.min(scan.boresight,0)+np.min(beam_offs,(0,1)),
				np.max(scan.boresight,0)+np.max(beam_offs,(0,1))])
		# Build our pointing interpolator
		transform  = pos2pix(scan,template,sys)
		ipol, obox = build_interpol(transform, ibox, id=scan.entry.id)
		self.rbox, self.nbox, self.yvals = extract_interpol_params(ipol, template.dtype)
		# And store our data
		self.beam_offs, self.beam_comps = beam_offs, beam_comps
		self.pixbox, self.nphi  = build_pixbox(obox[:,:2], template)
		self.scan,   self.dtype = scan, template.dtype
		self.func  = get_core(self.dtype).pmat_map
		self.order = config.get("pmat_map_order", order)
	def forward(self, tod, m, tmul=1, mmul=1, times=None):
		"""m -> tod"""
		# Loop over each beam, summing its contributions
		if times is None: times = np.zeros(5)
		tod *= tmul
		for bi, (boff, bcomp) in enumerate(zip(self.beam_offs, self.beam_comps)):
			self.func( 1, 1, self.order+1, 1, mmul, tod.T, m.T, self.scan.boresight.T, self.scan.hwp_phase.T,
					boff.T, bcomp.T, self.rbox.T, self.nbox, self.yvals.T, self.pixbox.T, self.nphi, times)
	def backward(self, tod, m, tmul=1, mmul=1, times=None):
		"""tod -> m"""
		if times is None: times = np.zeros(5)
		m *= mmul
		for bi, (boff, bcomp) in enumerate(zip(self.beam_offs, self.beam_comps)):
			self.func(-1, 1, self.order+1, tmul, 1, tod.T, m.T, self.scan.boresight.T, self.scan.hwp_phase.T,
					boff.T, bcomp.T, self.rbox.T, self.nbox, self.yvals.T, self.pixbox.T, self.nphi, times)

def get_moby_pointing(entry, bore, dets, downgrade=1):
	# Set up moby2
	import moby2
	moby2.pointing.set_bulletin_A()
	params = {
		"detector_offsets": {
			"format": "fp_file",
			"filename": entry.point_template },
		"pol_source": {
			"source": "file",
			"filename": entry.polangle,
			"format": "ascii",
			"fail_value": -1 },
		"shift_generator": {
			"source": "file",
			"filename": entry.point_offsets,
			"columns": [0,5,6],
			"rescale_degrees": 180/np.pi},
		}
	bore   = np.ascontiguousarray(bore[::downgrade].T)
	fplane = moby2.scripting.get_focal_plane(params, det_uid=dets, tod=entry.id)
	res = fplane.get_coords(bore[0], bore[1], bore[2], fields=['ra','dec','cos_2gamma','sin_2gamma'])
	res[0][:] = utils.unwind(res[0])
	return np.array(res)

downsamp = config.default("pmat_moby_downsamp", 20, "How much to downsample pointing by in pmat moby when fitting model")
class PmatMoby(PointingMatrix):
	def __init__(self, scan, template, sys=None):
		sys      = config.get("map_sys",      sys)
		downsamp = config.get("pmat_moby_downsamp", 20)

		bore = scan.boresight.copy()
		bore[:,0] += utils.mjd2ctime(scan.mjd0)
		opoint = get_moby_pointing(scan.entry, bore, scan.dets, downgrade=downsamp)
		# We will fit a polynomial to the pointing for each detector
		box = np.array([[np.min(a),np.max(a)] for a in scan.boresight.T]).T
		def scale(a, r): return 2*(a-r[0])/(r[1]-r[0])-1
		t  = scale(scan.boresight[::downsamp,0], box[:,0])
		az = scale(scan.boresight[::downsamp,1], box[:,1])
		el = scale(scan.boresight[::downsamp,2], box[:,2])

		basis  = np.array([az**4, az**3, az**2, az**1, az**0, el**2, el, t**2, t, t*az]).T
		denom  = basis.T.dot(basis)
		e,v = np.linalg.eigh(denom)
		if (np.min(e) < 1e-8*np.max(e)):
			basis = np.concatenate([basis[:,:5],basis[:,6:]],1)
			denom = basis.T.dot(basis)
		# Convert from ra/dec to pixels, since we've confirmed that
		# our ra/dec is the same as ninkasi to 0.01".
		t1 = time.time()
		pix= template.sky2pix(opoint[1::-1],safe=True)
		t2 = time.time()
		#rafit  = np.linalg.solve(denom, basis.T.dot(opoint[0].T))
		#decfit = np.linalg.solve(denom, basis.T.dot(opoint[1].T))
		yfit = np.linalg.solve(denom, basis.T.dot(pix[0].T))
		xfit = np.linalg.solve(denom, basis.T.dot(pix[1].T))

		# Just use the same az and t as before for simplicity. The
		# coefficients will be different, but the result will be
		# the same.
		basis = np.array([az**0, az**1, az**2, az**3, t]).T
		denom = basis.T.dot(basis)

		cosfit = np.linalg.solve(denom, basis.T.dot(opoint[2].T))
		sinfit = np.linalg.solve(denom, basis.T.dot(opoint[3].T))

		# Parameters for pmat
		self.posfit = np.concatenate([yfit[:,:,None],xfit[:,:,None]],2)
		self.polfit = np.concatenate([cosfit[:,:,None],sinfit[:,:,None]],2)
		self.box    = box
		self.pixbox = np.array([[0,0],template.shape[-2:]])
		self.scan   = scan
		self.dtype  = template.dtype
		self.core   = get_core(self.dtype)
	def forward(self, tod, m, tmul=0, mmul=1):
		"""m -> tod"""
		self.core.pmat_ninkasi( 1, tmul, mmul, tod.T, m.T, self.scan.boresight.T, self.box.T, self.pixbox.T, self.posfit.T, self.polfit.T)
	def backward(self, tod, m, tmul=1, mmul=1):
		"""tod -> m"""
		self.core.pmat_ninkasi(-1, tmul, mmul, tod.T, m.T, self.scan.boresight.T, self.box.T, self.pixbox.T, self.posfit.T, self.polfit.T)
	def translate(self, bore=None, offs=None, comps=None):
		raise NotImplementedError

# Neither this approach nor time-domain linear interpolation works at the moment.
# In theory, they would help with the subpixel bias issue. In practice, they
# lead to even mjore subpixel bias, and runaway residuals. There are two issues
#  1. In order for CG to converge, forward must be the exact transpose of backward.
#  2. In order to get rid of the bias, forward must be a good approximation of the
#     beam, and must not have any pixel offsets or similar.

class PmatCut(PointingMatrix):
	"""Implementation of cuts-as-extra-degrees-of-freedom for a single
	scan."""
	def __init__(self, scan, params=None):
		params = config.get("pmat_cut_type", params)
		n, neach, flat = scan.cut.flatten()
		# Detectors for each cut
		dets = np.concatenate([np.zeros(n,dtype=int)+i for i,n in enumerate(neach)])
		# Extract the cut parameters. E.g. poly:foo_secs -> [4,foo_samps]
		par  = np.array(self.parse_params(params, scan.srate))
		# Meaning of cuts array: [:,{dets,offset,length,out_length,type,args..}]
		self.cuts = np.zeros([flat.shape[0],5+len(par)],dtype=np.int32)
		self.cuts[:,0] = dets
		self.cuts[:,1] = flat[:,0]
		self.cuts[:,2] = flat[:,1]-flat[:,0]
		# Set up the parameter arguments
		self.cuts[:,5:]= par[None,:]
		assert np.all(self.cuts[:,2] > 0),  "Empty cut range detected in %s" % scan.entry.id
		assert np.all(self.cuts[:,1] >= 0) and np.all(flat[:,1] <= scan.nsamp), "Out of bounds cut range detected in %s" % scan.entry.id
		if self.cuts.size > 0:
			get_core(np.float32).measure_cuts(self.cuts.T)
		self.cuts[:,3] = utils.cumsum(self.cuts[:,4])
		# njunk is the number of cut parameters for *this scan*
		self.njunk  = np.sum(self.cuts[:,4])
		self.params = params
		self.scan = scan
	def forward(self, tod, junk):
		"""Project from the cut parameter (junk) space for this scan
		to tod."""
		if self.cuts.size > 0:
			get_core(tod.dtype).pmat_cut( 1, tod.T, junk, self.cuts.T)
	def backward(self, tod, junk):
		"""Project from tod to cut parameters (junk) for this scan.
		This is meant to be called before the map projection, and
		removes the cut samples from the tod at the same time,
		replacing them with zeros. That way the map projection can
		be done without needing to care about the cuts."""
		if self.cuts.size > 0:
			get_core(tod.dtype).pmat_cut(-1, tod.T, junk, self.cuts.T)
	def parse_params(self,params,srate):
		toks = params.split(":")
		kind = toks[0]
		args = [float(s) for s in toks[1].split(",")] if len(toks) > 1 else []
		# Transform from seconds to samples if needed
		if kind in ["bin","exp","poly"]: args[0] = args[0]*srate+0.5
		return [{"none":0,"full":1,"bin":2,"exp":3,"poly":4}[kind]]+[int(arg) for arg in args]

config.default("pmat_parallax_au", 0, "Sun distance to use for parallax correction in pointing matrices, in AU. 0 disables parallax.")
class pos2pix:
	"""Transforms from scan coordinates to pixel-center coordinates.
	This becomes discontinuous for scans that wrap from one side of the
	sky to another for full-sky pixelizations."""
	def __init__(self, scan, template, sys, ref_phi=0):
		self.scan, self.template, self.sys = scan, template, sys
		self.ref_phi = ref_phi
	def __call__(self, ipos):
		"""Transform ipos[{t,az,el},nsamp] into opix[{y,x,c,s},nsamp]."""
		shape = ipos.shape[1:]
		ipos  = ipos.reshape(ipos.shape[0],-1)
		time  = self.scan.mjd0 + ipos[0]/utils.day2sec
		opos = coordinates.transform(self.scan.sys, self.sys, ipos[1:], time=time, site=self.scan.site, pol=True)
		# Parallax correction
		sundist = config.get("pmat_parallax_au")
		if sundist:
			# Transform to a sun-centered coordinate system, assuming all objects
			# are at a distance of sundist from the sun
			opos[1::-1] = parallax.earth2sun(opos[1::-1], self.scan.mjd0, sundist)

		opix = np.zeros((4,)+ipos.shape[1:])
		if self.template is not None:
			opix[:2] = self.template.sky2pix(opos[1::-1],safe=2)
			# When mapping the full sky, angle wraps can't be hidden
			# ouside the image. We must therefore unwind along each
			# interpolation axis to avoid discontinuous interpolation
			nx = int(np.abs(360/self.template.wcs.wcs.cdelt[0]))
			opix[1] = utils.unwind(opix[1].reshape(shape), period=nx, axes=range(len(shape))).reshape(-1)
		else:
			# If we have no template, output angles instead of pixels.
			# Make sure the angles don't have any jumps in them
			opix[:2] = opos[1::-1] # output order is dec,ra
			opix[1]  = utils.rewind(opix[1], self.ref_phi)
		opix[2]  = np.cos(2*opos[2])
		opix[3]  = np.sin(2*opos[2])
		return opix.reshape((opix.shape[0],)+shape)

config.default("pmat_ptsrc_rsigma", 4.0, "Max number of standard deviations away from a point source to compute the beam profile. Larger values are slower but more accurate.")
class PmatPtsrc(PointingMatrix):
	def __init__(self, scan, params, sys=None, tmul=None, pmul=None):
		# Params is [nsrc,{dec,ra,amps,ibeams}]
		rmul  = config.get("pmat_ptsrc_rsigma")
		self.dtype = params.dtype
		transform  = build_pos_transform(scan, sys=config.get("map_sys", sys))
		ipol, obox = build_interpol(transform, scan.box, scan.entry.id, posunit=0.1*utils.arcsec)
		# It's error prone to require the user to have the angles consistently
		# wrapped. So we will rewrap params ourselves
		self.ref = np.mean(obox[:,:2],0)
		params   = params.copy()
		params[:2] = utils.rewind(params[:2].T,self.ref).T

		self.rbox, self.nbox, self.yvals = extract_interpol_params(ipol, self.dtype)
		self.comps = np.arange(params.shape[0]-5)
		self.scan  = scan
		self.core = pmat_core_32.pmat_core if self.dtype == np.float32 else pmat_core_64.pmat_core

		# Collect information about which samples hit which sources
		nsrc = params.shape[1]
		ndet = scan.ndet
		# rhit is the inverse of the squared amplitude-weighted inverse beam for
		# some reason. But it is basically going to be our beam size.
		print "FIXME: Point source format incompatible with PmatPtsrc"
		# To be precise, this function assumes that point sources are [param,nsrc]
		# rathar than [nsrc,param], that PmatPtsrc2 assumes. I should decide which
		# one I want. The whole point source stuff is messy at the moment.

		rhit = np.zeros(nsrc)+(np.sum(1./params[-3]*params[2]**2)/np.sum(params[2]**2))**0.5
		rmax = rhit*rmul
		try: det_ivars = scan.noise.ivar
		except NotImplementedError: det_ivars = np.zeros(ndet)
		src_ivars = np.zeros(nsrc,dtype=self.dtype)

		# Measure ranges. May need to iterate if initial allocation was too small
		nrange = np.zeros([nsrc,ndet],dtype=np.int32)
		ranges = np.zeros([nsrc,ndet,100,2],dtype=np.int32)
		self.core.pmat_ptsrc_prepare(params, rhit, rmax, det_ivars, src_ivars, ranges.T, nrange.T, self.scan.boresight.T, self.scan.offsets.T, self.rbox.T, self.nbox, self.yvals.T)
		if np.max(nrange) > ranges.shape[2]:
			ranges = np.zeros([nsrc,ndet,np.max(nrange),2],dtype=np.int32)
			self.core.pmat_ptsrc_prepare(params, rhit, rmax, det_ivars, src_ivars, ranges.T, nrange.T, self.scan.boresight.T, self.scan.offsets.T, self.rbox.T, self.nbox, self.yvals.T)
		self.ranges, self.rangesets, self.offsets = compress_ranges(ranges, nrange, scan.cut, scan.nsamp)
		self.src_ivars = src_ivars
		self.nhit = np.sum(self.ranges[:,1]-self.ranges[:,0])

		self.tmul = 0 if tmul is None else tmul
		self.pmul = 1 if pmul is None else pmul

	def forward(self, tod, params, tmul=None, pmul=None):
		"""params -> tod"""
		if tmul is None: tmul = self.tmul
		if pmul is None: pmul = self.pmul
		pcopy = params.copy()
		pcopy[:2] = utils.rewind(pcopy[:2].T,self.ref).T
		self.core.pmat_ptsrc(tmul, pmul, tod.T, pcopy, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.yvals.T, self.ranges.T, self.rangesets, self.offsets.T)

	def extract(self, tod, cut=None, raw=0):
		"""Extract precomputed pointing and phase information for the selected samples.
		These are stored in a (somewhat cumbersome) compressed format, where sample I
		of range R of detector D and source S has index ranges[rangesets[offsets[S,D,0]+R],0]+I.
		The reason for this complicated setup is to store each TOD sample only once, even if
		it hits multiple sources. This is not primarily to save space, but to be able to
		perform likelihood analysis on it. Subtracting one source from one copy of a sample,
		and another source from another copy of the sample, will leave each copy with some
		signal left unmodeled.

		If cut is specified, the response of cut samples will be set to zero, giving them
		no weight.

		If raw is a nonzero integer, the output pointing and phase will be in the
		telescope's native coordinates (i.e. hor) rather than the normal output coordinates.
		"""
		point = np.zeros([self.nhit,2+(raw>0)],dtype=self.dtype)
		phase = np.zeros([self.nhit,len(self.comps)],dtype=self.dtype)
		srctod= np.zeros([self.nhit],dtype=self.dtype)
		oranges= np.zeros(self.ranges.shape, dtype=np.int32)
		self.core.pmat_ptsrc_extract(tod.T, srctod, point.T, phase.T, oranges.T, self.scan.boresight.T,
				self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox,
				self.yvals.T, self.ranges.T, self.rangesets, self.offsets.T, raw)
		if cut:
			# Cuts are handled by setting the phase (response) to zero
			mtod = cut.to_mask().astype(self.dtype)
			mask = np.zeros([self.nhit],dtype=self.dtype)
			self.core.pmat_ptsrc_extract(mtod.T, mask, point.T, phase.T, oranges.T, self.scan.boresight.T,
					self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox,
					self.yvals.T, self.ranges.T, self.rangesets, self.offsets.T, raw)
			mask = np.rint(mask)==1
			phase[mask] = 0
		# Store the raw pointing offsets, so we can ensure that the point fit does everything the same
		# way the main mapmaker does.
		mean_point = np.mean(self.scan.boresight.T[1:],1)
		raw_offsets = coordinates.recenter(mean_point[:,None] + self.scan.offsets.T[1:], np.concatenate([mean_point,mean_point*0])).T
		res = bunch.Bunch(point=point, phase=phase, tod=srctod, ranges=oranges, rangesets=self.rangesets, offsets=self.offsets, dets=self.scan.dets, rbox=self.rbox, nbox=self.nbox, yvals=self.yvals, point_offset=raw_offsets, ivars=np.ones(len(self.scan.dets)))
		return res

config.default("pmat_ptsrc_cell_res", 5, "Cell size in arcmin to use for fast source lookup.")
class PmatPtsrc2(PointingMatrix):
	def __init__(self, scan, srcs, sys=None, tmul=None, pmul=None):
		# We support a srcs which is either [nsrc,nparam] or [nsrc,ndir,nparam], where
		# ndir is either 1 or 2 depending on whether one wants to separate different
		# scanning directions.
		srcs = np.array(srcs)
		if srcs.ndim == 2: srcs = srcs[:,None]
		# srcs is [ndir,nsrc,{dec,ra,T,Q,U,ibeams}]
		sys   = config.get("map_sys", sys)
		cres  = config.get("pmat_ptsrc_cell_res")*utils.arcmin
		ndir       = srcs.shape[1]
		self.scan  = scan
		maxcell    = 50 # max numer of sources per cell

		# Compute parallax displacement if necessary
		sundist = config.get("pmat_parallax_au")
		self.dpos = 0
		if sundist:
			# Transformation to a sun-centered system
			self.dpos = parallax.earth2sun(srcs.T[:2], self.scan.mjd0, sundist, diff=True).T
		srcs[:,:,:2] += self.dpos

		print "srcs", srcs

		# Investigate the beam to find the max relevant radius
		sigma_lim = config.get("pmat_ptsrc_rsigma")
		value_lim = np.exp(-0.5*sigma_lim**2)
		rmax = np.where(scan.beam[1]>=value_lim)[0][-1]*scan.beam[0,1]
		rmul = max([utils.expand_beam(src[-3:])[0][0] for src in srcs.reshape(-1,srcs.shape[-1])])
		rmax *= rmul

		# Build interpolator (dec,ra output ordering)
		transform  = build_pos_transform(scan, sys=config.get("map_sys", sys))
		ipol, obox = build_interpol(transform, scan.box, scan.entry.id, posunit=0.1*utils.arcsec)
		self.rbox, self.nbox, self.yvals = extract_interpol_params(ipol, srcs.dtype)

		# Build source hit grid
		cbox    = obox[:,:2]
		cshape  = tuple(np.ceil(((cbox[1]-cbox[0])/cres)).astype(int))
		self.ref = np.mean(cbox,0)
		srcs[:,:,:2] = utils.rewind(srcs[:,:,:2], self.ref)

		print "cbox", cbox/utils.degree
		print "cshape", cshape

		# A cell is hit if it overlaps both horizontall any vertically
		# with the point source +- rmax
		ncell = np.zeros((ndir,)+cshape,dtype=np.int32)
		cells = np.zeros((ndir,)+cshape+(maxcell,),dtype=np.int32)
		c0 = cbox[0]; inv_dc = cshape/(cbox[1]-cbox[0])
		for si, dsrc in enumerate(srcs):
			for sdir, src in enumerate(dsrc):
				i1 = (src[:2]-rmax-c0)*inv_dc
				i2 = (src[:2]+rmax-c0)*inv_dc+1 # +1 because this is a half-open interval
				# Truncate to edges - any source outside of our region
				# will be put on one of the edge cells
				i1 = np.maximum(i1.astype(int), 0)
				i2 = np.minimum(i2.astype(int), np.array(cshape)-1)
				print si, sdir, i1, i2, cshape
				if np.any(i1 >= cshape) or np.any(i2 < 0): continue
				sel= (sdir,slice(i1[0],i2[0]),slice(i1[1],i2[1]))
				cells[sel][:,:,ncell[sel]] = si
				ncell[sel] += 1
		self.cells, self.ncell = cells, ncell
		self.rmax = rmax
		self.cbox = cbox
		self.tmul = 1 if tmul is None else tmul
		self.pmul = 1 if pmul is None else pmul
	def apply(self, dir, tod, srcs, tmul=None, pmul=None):
		if tmul is None: tmul = self.tmul
		if pmul is None: pmul = self.pmul
		if srcs.ndim == 2: srcs = srcs[:,None]
		# Handle angle wrapping without modifying the original srcs array
		wsrcs = srcs.copy()
		wsrcs[:,:,:2] = utils.rewind(srcs[:,:,:2], self.ref) + self.dpos
		t1 = time.time()
		core = get_core(tod.dtype)
		core.pmat_ptsrc2(dir, tmul, pmul, tod.T, wsrcs.T,
				self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T,
				self.rbox.T, self.nbox.T, self.yvals.T,
				self.scan.beam[1], self.scan.beam[0,-1], self.rmax,
				self.cells.T, self.ncell.T, self.cbox.T)
		# Copy out any amplitudes that may have changed
		srcs[:,:,2:5] = wsrcs[:,:,2:5]
	def forward(self, tod, srcs, tmul=None, pmul=None):
		"""srcs -> tod"""
		self.apply( 1, tod, srcs, tmul=tmul, pmul=pmul)
	def backward(self, tod, srcs, tmul=None, pmul=None):
		"""tod -> srcs"""
		self.apply(-1, tod, srcs, tmul=tmul, pmul=pmul)

def build_interpol(transform, box, id="none", posunit=1.0, sys=None):
	sys   = config.get("map_sys",      sys)
	# We widen the bounding box slightly to avoid samples falling outside it
	# due to rounding errors.
	box = utils.widen_box(np.array(box), 1e-3)
	acc = config.get("pmat_accuracy")
	ip_size = config.get("pmat_interpol_max_size")
	ip_time = config.get("pmat_interpol_max_time")

	# Build pointing interpolator
	errlim = np.array([1e-3*posunit,1e-3*posunit,utils.arcmin,utils.arcmin])*acc
	ipol, obox, ok, err = interpol.build(transform, interpol.ip_linear, box, errlim, maxsize=ip_size, maxtime=ip_time, return_obox=True, return_status=True)
	if not ok: print "Warning: Accuracy %g was specified, but only reached %g for tod %s" % (acc, np.max(err/errlim)*acc, id)
	return ipol, obox

def build_pos_transform(scan, sys):
	# Set up pointing interpolation
	box = np.array(scan.box)
	margin = (box[1]-box[0])*1e-3 # margin to avoid rounding erros
	margin[1:] += 10*utils.arcmin
	box[0] -= margin/2; box[1] += margin/2
	# Find the rough position of our scan
	ref_phi = coordinates.transform(scan.sys, sys, scan.box[:1,1:].T, time=scan.mjd0+scan.box[:1,0], site=scan.site)[0,0]
	return pos2pix(scan,None,sys,ref_phi=ref_phi)

class PmatScan(PointingMatrix):
	"""Project between tod and per-det az basis. Will be plain az
	basis if used with 2d maps. For [2,ny,nx]-maps, will fill one
	component with rightgoing scans and one with leftgoing scans."""
	def __init__(self, scan, area, dets):
		abox = area.box()[:,1]
		self.scan, self.az0, self.daz = scan, abox[0], (abox[1]-abox[0])/area.shape[-1]
		self.az, self.dets = utils.rewind(self.scan.boresight[:,1],0), dets
	def apply(self, tod, m, dir):
		core = get_core(tod.dtype)
		if m.ndim == 3 and m.shape[0] == 2:
			core.pmat_phase(dir, tod.T, m.T, self.az, self.dets, self.az0, self.daz)
		elif m.ndim == 2:
			core.pmat_az   (dir, tod.T, m.T, self.az, self.dets, self.az0, self.daz)
		else: raise ValueError("PmatScan needs a [ny,nx] or [2,ny,nx] map")
	def forward(self, tod, m):  self.apply(tod, m, 1)
	def backward(self, tod, m): self.apply(tod, m,-1)

class PmatPhaseFlat(PointingMatrix):
	def __init__(self, az, a0, daz, naz):
		self.az = az
		self.a0, self.daz, self.naz = a0, daz, naz
	def apply(self, tod, phase, dir, tmul=1):
		map  = np.zeros((2,len(tod),self.naz),dtype=phase.dtype)
		if dir > 0:
			map[:] = phase[:,None,:]
			if tmul != 1: tod[:] *= tmul
		pmat_phase(dir, tod, map, self.az, np.arange(len(tod)), self.a0, self.daz)
		if dir < 0: phase[:] = np.sum(map,1)
	def forward(self, tod, phase, tmul=1): return self.apply(tod, phase, 1, tmul=tmul)
	def backward(self,tod, phase, tmul=1): return self.apply(tod, phase,-1, tmul=tmul)

def compress_ranges(ranges, nrange, cut, nsamp):
	"""Given ranges[nsrc,ndet,nmax,2], nrange[nsrc,ndet] where ranges has
	det-local numbering, return the same information in a compressed format
	ranges[nr,2], rangesets[nind], offsets[nsrc,ndet,2], where ranges still has
	per-detector ordering. It used to be in global sample ordering, but I always
	ended up converting back afterwards."""
	nsrc, ndet = nrange.shape
	# Special case: None hit. We represent this as a single range hitting no samples,
	# which isn't used by any of the srcs.
	def dummy():
		ranges  = np.array([[0,0]],dtype=np.int32)
		rangesets = np.array([0],dtype=np.int32)
		offsets = np.zeros([nsrc,ndet,2],dtype=np.int32)
		return ranges, rangesets, offsets
	if np.sum(nrange) == 0: return dummy()
	# First collapse ranges,nrange to flat ranges and indices into it
	det_ranges = []
	maps       = []
	nflat      = 0
	offsets    = np.zeros([nsrc,ndet,2],dtype=np.int32)
	for di in xrange(ndet):
		# Collect the sample ranges for all the sources for a given detector
		src_ranges = []
		for si in xrange(nsrc):
			# Offsets holds the indices to the first and last+1 range for each
			# source and detector. We get this simply by counting how many ranges
			# we have processed so far. After merging, these will be indices into
			# the map array instead.
			offsets[si,di,0] = nflat
			if nrange[si,di] > 0:
				current_ranges  = ranges[si,di,:nrange[si,di]]
				cutsplit_ranges = utils.range_sub(current_ranges, cut[di].ranges)
				nflat += len(cutsplit_ranges)
				if len(cutsplit_ranges) > 0:
					src_ranges.append(cutsplit_ranges)
			offsets[si,di,1] = nflat
		if len(src_ranges) > 0:
			src_ranges = np.concatenate(src_ranges)
			# Merge overlapping ranges for this detector. Map maps from
			# indices into the unmerged array to indices into the merged array.
			# We merge at this step rather than at the end to avoid merging
			# samples from one detector with samples from the next.
			src_merged, map = utils.range_union(src_ranges, mapping=True)
			det_ranges.append(src_merged)
			maps.append(map)
	# Concatenate the detector ranges into one long list. Make sure
	# that we actually have some ranges left. While we did check at the
	# start, the cuts may have eliminated the ranges we started with.
	if sum([len(r) for r in det_ranges]) == 0: return dummy()
	oranges = np.concatenate(det_ranges)
	moffs   = utils.cumsum([len(r) for r in det_ranges])
	map     = np.concatenate([m+o for m,o in zip(maps,moffs)])
	return oranges, map, offsets

def extract_interpol_params(ipol, dtype):
	"""Extracts flattend interpolation parameters from an Interpolator object
	in a form suitable for passing to fortran. Returns rbox[{from,to},nparam],
	nbox[nparam] (grid size along each input parameter)"""
	rbox  = ipol.box
	nbox  = np.array(ipol.y.shape[1:])
	yvals = np.ascontiguousarray(ipol.y.reshape(ipol.y.shape[0],-1).T)
	return rbox, nbox, yvals

def build_pixbox(obox, template, margin=10):
	# We allow negative and positive overshoot to allow the
	# pointing matrix to handle pixel wraparound.
	res = np.array([np.floor(obox[0]-margin),np.floor(obox[1]+margin)]).astype(np.int32)
	# If the original box extends outside [[0,0],n], this box may
	# have empty ranges. If so, return a single-pixel box
	if np.any(res[1]<=res[0]):
		res = np.array([[0,0],[1,1]],dtype=np.int32)
	nphi = int(np.abs(360./template.wcs.wcs.cdelt[0]))
	return res, nphi

def pmat_phase(dir, tod, map, az, dets, az0, daz):
	core = get_core(tod.dtype)
	core.pmat_phase(dir, tod.T, map.T, az, dets, az0, daz)

def pmat_plain(dir, tod, map, pix):
	core = get_core(tod.dtype)
	core.pmat_plain(dir, tod.T, map.T, pix.T)

