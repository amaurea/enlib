"""This module contains implementations for "pointing matrices", which project
between time-ordered data (described by a Scan class) down to some other space,
such as a map. Its general interface is .forward(tod,m) and .backward(tod,m).
Both of these are allowed to modify both arguments, though usually
forward will update tod based on m, and backward will update m based on tod.
The reason for allowing the other argument to be modified is to make it easier
to incrementally project different parts of the signal.
"""
import numpy as np, bunch, time
from enlib import enmap, interpol, utils, coordinates, config, errors, array_ops
import pmat_core_32
import pmat_core_64
def get_core(dtype):
	if dtype == np.float32:
		return pmat_core_32.pmat_core
	else:
		return pmat_core_64.pmat_core

config.default("pmat_map_order",      0, "The interpolation order of the map pointing matrix.")
config.default("pmat_cut_type",  "full", "The cut sample representation used. 'full' uses one degree of freedom for each cut sample. 'bin:N' uses one degree of freedom for every N samples. 'exp' used one degree of freedom for the first sample, then one for the next two, one for the next 4, and so on, giving high resoultion at the edges of each cut range, and low resolution in the middle.")
config.default("map_eqsys",       "equ", "The coordinate system of the maps. Can be eg. 'hor', 'equ' or 'gal'.")
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
		sys   = config.get("map_eqsys",      sys)
		order = config.get("pmat_map_order", order)

		# We widen the bounding box slightly to avoid samples falling outside it
		# due to rounding errors.
		box = utils.widen_box(np.array(scan.box), 1e-3)
		acc  = config.get("pmat_accuracy")
		ip_size= config.get("pmat_interpol_max_size")
		ip_time= config.get("pmat_interpol_max_time")
		transform = pos2pix(scan,template,sys)

		# Build pointing interpolator
		errlim = np.array([1e-2,1e-2,utils.arcmin,utils.arcmin])*0.1*acc
		ipol, obox, ok, err = interpol.build(transform, interpol.ip_linear, box, errlim, maxsize=ip_size, maxtime=ip_time, return_obox=True, return_status=True)
		if not ok: print "Warning: Accuracy %g was specified, but only reached %g for tod %s" % (acc, np.max(err/errlim)*acc, scan.entry.id)

		self.rbox, self.nbox, self.ys = extract_interpol_params(ipol, template.dtype)
		# Use obox to extract a pixel bounding box for this scan.
		# These are the only pixels pmat needs to concern itself with.
		# Reducing the number of pixels makes us more memory efficient
		self.pixbox = build_pixbox(obox[:,:2], template.shape[-2:])

		self.comps= np.arange(template.shape[0])
		self.scan  = scan
		self.order = order
		self.dtype = template.dtype
		self.core = get_core(self.dtype)
		if order == 0:
			self.func = self.core.pmat_nearest
		elif order == 1:
			self.func = self.core.pmat_linear
		else:
			raise NotImplementedError("order > 1 is not implemented")
		self.transform = transform
		self.ipol = ipol
	def forward(self, tod, m, tmul=1, mmul=1):
		"""m -> tod"""
		self.func( 1, tmul, mmul, tod.T, m.T, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T, self.pixbox.T)
	def backward(self, tod, m, tmul=1, mmul=1):
		"""tod -> m"""
		self.func(-1, tmul, mmul, tod.T, m.T, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T, self.pixbox.T)
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
		self.core.translate(bore.T, pix.T, phase.T, offs.T, comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T)
		return pix, phase

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
		sys      = config.get("map_eqsys",      sys)
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

		## For some reason pol angles are interpolated differently
		#t  = np.linspace(0,1,bore.shape[0])[::downsamp]
		#az = bore[::downsamp,1]
		#npoly = 3
		#basis = np.zeros([len(t),2+npoly])
		#basis[:,0]  = 1
		#basis[:,-1] = t
		#for i in range(npoly):
		#	basis[:,i+1] = basis[:,i]*az

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
#
#class PmatMapIP(PointingMatrix):
#	def __init__(self, scan, template, sys=None, order=None, oversample=5):
#		self.scan     = scan
#		self.template = template
#		self.dtype    = template.dtype
#		self.sys      = sys
#		self.oversample= oversample
#		ncomp, h, w = template.shape
#		wcs = template.wcs.deepcopy()
#		wcs.wcs.cdelt /= oversample
#		wcs.wcs.crpix *= oversample
#		self.big = enmap.zeros([ncomp,h*oversample,w*oversample],wcs=wcs,dtype=self.dtype)
#		self.pmat = PmatMap(scan, self.big, self.sys, order)
#	def forward(self, tod, m):
#		# Interpolate map to full res
#		print "FA", np.sum(m**2)
#		m2 = m.project(self.big.shape, self.big.wcs)
#		self.pmat.forward(tod, m2)
#		print "FB", np.sum(tod**2)
#	def backward(self, tod, m):
#		print "BA", np.sum(tod**2)
#		self.big[...] = 0
#		self.pmat.backward(tod, self.big)
#		# This is not the real transpose of project.
#		m[...] = enmap.downgrade(self.big, self.oversample)
#		print "Bb", np.sum(m**2)
#	def translate(self, bore=None, offs=None, comps=None):
#		return self.pmat.translate(bore, offs, comps)

class PmatMapSlow(PointingMatrix):
	"""Reference implementation of the simple nearest neighbor
	pointing matrix. Very slow - not meant for serious use. It's interesting
	as an example of how a reasonable python-only implementation would
	perform. Uses trilinear interpolation for coordinates."""
	def __init__(self, scan, template, sys=None):
		sys = config.get("map_eqsys", sys)
		# Build interpolator between scan coordinates and
		# template map coordinates.
		self.ipol = interpol.build(pos2pix(scan,template,sys), interpol.ip_linear, scan.box, [1e-3,1e-3,utils.arcsec,utils.arcsec], order=1)
		self.scan = scan
	def get_pix_phase(self,d):
		pix   = self.ipol(self.scan.boresight.T+self.scan.offsets[d,:,None])
		ipix  = np.floor(pix[:2]).astype(np.int32)
		phase = np.zeros([self.scan.comps.shape[1],self.scan.boresight.shape[0]])
		phase[...] = self.scan.comps[d,:,None]
		phase[1] = self.scan.comps[d,1] * pix[2] + self.scan.comps[d,2] * pix[3]
		phase[2] = self.scan.comps[d,1] * pix[3] - self.scan.comps[d,2] * pix[2]
		return ipix, phase
	def backward(self, tod, m):
		for d in range(tod.shape[0]):
			pix, phase = self.get_pix_phase(d)
			for c in range(m.shape[0]):
				m[c] += np.bincount(np.ravel_multi_index(pix[:2], m.shape[-2:]), tod[d], m[c].size).reshape(m[c].shape)
	def forward(self, tod, m):
		for d in range(tod.shape[0]):
			pix, phase = self.get_pix_phase(d)
			tod[d] += np.sum(m[:,pix[0],pix[1]]*phase[:m.shape[0]],0)

class PmatCut(PointingMatrix):
	"""Implementation of cuts-as-extra-degrees-of-freedom for a single
	scan."""
	def __init__(self, scan, params=None):
		params = config.get("pmat_cut_type", params)
		n, neach, flat = scan.cut.flatten()
		dets = np.concatenate([np.zeros(n,dtype=int)+i for i,n in enumerate(neach)])
		par  = np.array(self.parse_params(params))
		self.cuts = np.zeros([flat.shape[0],5+len(par)],dtype=np.int32)
		self.cuts[:,0] = dets
		self.cuts[:,1] = flat[:,0]
		self.cuts[:,2] = flat[:,1]-flat[:,0]
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
	def parse_params(self,params):
		toks = params.split(":")
		kind = toks[0]
		args = tuple([int(s) for s in toks[1].split(",")]) if len(toks) > 1 else ()
		return ({"none":0,"full":1,"bin":2,"exp":3,"poly":4}[toks[0]],)+args

class pos2pix:
	"""Transforms from scan coordinates to pixel-center coordinates."""
	def __init__(self, scan, template, sys, ref_phi=0):
		self.scan, self.template, self.sys = scan, template, sys
		self.ref_phi = ref_phi
	def __call__(self, ipos):
		shape = ipos.shape[1:]
		ipos  = ipos.reshape(ipos.shape[0],-1)
		time  = self.scan.mjd0 + ipos[0]/utils.day2sec
		opos = coordinates.transform(self.scan.sys, self.sys, ipos[1:], time=time, site=self.scan.site, pol=True)
		opix = np.zeros((4,)+ipos.shape[1:])
		if self.template is not None:
			opix[:2] = self.template.sky2pix(opos[1::-1],safe=True)
		else:
			# If we have no template, output angles instead of pixels.
			# Make sure the angles don't have any jumps in them
			opix[:2] = opos[1::-1] # output order is dec,ra
			opix[1]  = utils.rewind(opix[1], self.ref_phi)
		opix[2]  = np.cos(2*opos[2])
		opix[3]  = np.sin(2*opos[2])
		return opix.reshape((opix.shape[0],)+shape)

class PmatMapRebin(PointingMatrix):
	"""Fortran-accelerated rebinning of maps."""
	def forward (self, mhigh, mlow):
		get_core(mhigh.dtype).pmat_map_rebin( 1, mhigh.T, mlow.T)
	def backward(self, mhigh, mlow):
		get_core(mhigh.dtype).pmat_map_rebin(-1, mhigh.T, mlow.T)

class PmatCutRebin(PointingMatrix):
	"""Fortran-accelerated rebinning of cut data."""
	def __init__(self, pmat_cut_high, pmat_cut_low):
		self.cut_high, self.cut_low = pmat_cut_high.cuts, pmat_cut_low.cuts
	def forward (self, jhigh, jlow):
		get_core(jhigh.dtype).pmat_cut_rebin( 1, jhigh.T, self.cut_high.T, jlow.T, self.cut_low.T)
	def backward(self, jhigh, jlow):
		get_core(jhigh.dtype).pmat_cut_rebin(-1, jhigh.T, self.cut_high.T, jlow.T, self.cut_low.T)

config.default("pmat_ptsrc_rsigma", 4.0, "Max number of standard deviations away from a point source to compute the beam profile. Larger values are slower but more accurate.")
class PmatPtsrc(PointingMatrix):
	def __init__(self, scan, params, sys=None, tmul=None, pmul=None):
		# Params is [nsrc,{dec,ra,amps,ibeams}]
		rmul  = config.get("pmat_ptsrc_rsigma")
		self.dtype = params.dtype
		ipol, obox = build_pos_interpol(scan, sys=config.get("map_eqsys", sys))
		# It's error prone to require the user to have the angles consistently
		# wrapped. So we will rewrap params ourselves
		self.ref = np.mean(obox[:,:2],0)
		params   = params.copy()
		params[:2] = utils.rewind(params[:2].T,self.ref).T

		self.rbox, self.nbox, self.ys = extract_interpol_params(ipol, self.dtype)
		self.comps = np.arange(params.shape[0]-5)
		self.scan  = scan
		self.core = pmat_core_32.pmat_core if self.dtype == np.float32 else pmat_core_64.pmat_core

		# Collect information about which samples hit which sources
		nsrc = params.shape[1]
		ndet = scan.ndet
		# rhit is the inverse of the squared amplitude-weighted inverse beam for
		# some reason. But it is basically going to be our beam size.
		print "FIXME: Point source format incompatible with PmatPtsrc"
		rhit = np.zeros(nsrc)+(np.sum(1./params[-3]*params[2]**2)/np.sum(params[2]**2))**0.5
		rmax = rhit*rmul
		try: det_ivars = scan.noise.ivar
		except NotImplementedError: det_ivars = np.zeros(ndet)
		src_ivars = np.zeros(nsrc,dtype=self.dtype)

		# Measure ranges. May need to iterate if initial allocation was too small
		nrange = np.zeros([nsrc,ndet],dtype=np.int32)
		ranges = np.zeros([nsrc,ndet,100,2],dtype=np.int32)
		self.core.pmat_ptsrc_prepare(params, rhit, rmax, det_ivars, src_ivars, ranges.T, nrange.T, self.scan.boresight.T, self.scan.offsets.T, self.rbox.T, self.nbox, self.ys.T)
		if np.max(nrange) > ranges.shape[2]:
			ranges = np.zeros([nsrc,ndet,np.max(nrange),2],dtype=np.int32)
			self.core.pmat_ptsrc_prepare(params, rhit, rmax, det_ivars, src_ivars, ranges.T, nrange.T, self.scan.boresight.T, self.scan.offsets.T, self.rbox.T, self.nbox, self.ys.T)
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
		self.core.pmat_ptsrc(tmul, pmul, tod.T, pcopy, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T, self.ranges.T, self.rangesets, self.offsets.T)

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
				self.ys.T, self.ranges.T, self.rangesets, self.offsets.T, raw)
		if cut:
			# Cuts are handled by setting the phase (response) to zero
			mtod = cut.to_mask().astype(self.dtype)
			mask = np.zeros([self.nhit],dtype=self.dtype)
			self.core.pmat_ptsrc_extract(mtod.T, mask, point.T, phase.T, oranges.T, self.scan.boresight.T,
					self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox,
					self.ys.T, self.ranges.T, self.rangesets, self.offsets.T, raw)
			mask = np.rint(mask)==1
			phase[mask] = 0
		# Store the raw pointing offsets, so we can ensure that the point fit does everything the same
		# way the main mapmaker does.
		mean_point = np.mean(self.scan.boresight.T[1:],1)
		raw_offsets = coordinates.recenter(mean_point[:,None] + self.scan.offsets.T[1:], np.concatenate([mean_point,mean_point*0])).T
		res = bunch.Bunch(point=point, phase=phase, tod=srctod, ranges=oranges, rangesets=self.rangesets, offsets=self.offsets, dets=self.scan.dets, rbox=self.rbox, nbox=self.nbox, ys=self.ys, point_offset=raw_offsets, ivars=np.ones(len(self.scan.dets)))
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
		sys   = config.get("map_eqsys", sys)
		cres  = config.get("pmat_ptsrc_cell_res")*utils.arcmin
		self.dtype = srcs.dtype
		ndir       = srcs.shape[1]
		self.core  = get_core(self.dtype)
		self.scan  = scan
		maxcell    = 10 # max numer of sources per cell

		# Investigate the beam to find the max relevant radius
		sigma_lim = config.get("pmat_ptsrc_rsigma")
		value_lim = np.exp(-0.5*sigma_lim**2)
		rmax = np.where(scan.beam[1]>=value_lim)[0][-1]*scan.beam[0,1]
		rmul = max([utils.expand_beam(src[-3:])[0][0] for src in srcs.reshape(-1,srcs.shape[-1])])
		rmax *= rmul

		# Build interpolator (dec,ra output ordering)
		ipol, obox = build_pos_interpol(scan, sys)
		self.rbox, self.nbox, self.ys = extract_interpol_params(ipol, self.dtype)
		# Build source hit grid
		cbox    = obox[:,:2]
		cshape  = tuple(((cbox[1]-cbox[0])/cres).astype(int))
		self.ref = np.mean(cbox,0)
		srcs[:,:,:2] = utils.rewind(srcs[:,:,:2], self.ref)
		# A cell is hit if it overlaps both horizontall any vertically
		# with the point source ± rmax
		ncell = np.zeros((ndir,)+cshape,dtype=np.int32)
		cells = np.zeros((ndir,)+cshape+(maxcell,),dtype=np.int32)
		for si, dsrc in enumerate(srcs):
			for sdir, src in enumerate(dsrc):
				i1 = np.maximum(0,((src[:2]-rmax-cbox[0])/cres).astype(int))
				i2 = np.minimum(cshape,((src[:2]+rmax-cbox[0])/cres).astype(int)+1)
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
		wsrcs[:,:,:2] = utils.rewind(srcs[:,:,:2], self.ref)
		t1 = time.time()
		self.core.pmat_ptsrc2(dir, tmul, pmul, tod.T, wsrcs.T,
				self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T,
				self.rbox.T, self.nbox.T, self.ys.T,
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

def build_pos_interpol(scan, sys):
	# Set up pointing interpolation
	box = np.array(scan.box)
	margin = (box[1]-box[0])*1e-3 # margin to avoid rounding erros
	margin[1:] += 10*utils.arcmin
	box[0] -= margin/2; box[1] += margin/2
	# Find the rough position of our scan
	ref_phi = coordinates.transform(scan.sys, sys, scan.box[:1,1:].T, time=scan.mjd0+scan.box[:1,0], site=scan.site)[0,0]
	#ref_phi = params[1,0]
	acc  = config.get("pmat_accuracy")
	ip_size= config.get("pmat_interpol_max_size")
	ip_time= config.get("pmat_interpol_max_time")
	transform = pos2pix(scan,None,sys,ref_phi=ref_phi)
	# With acc=1, this seems to achieve 0.015" accuracy, which is 10 times worse than what one would
	# naively expect.
	ipol, obox, ok, err = interpol.build(transform, interpol.ip_linear, box, np.array([0.01*utils.arcsec, 0.01*utils.arcsec ,utils.arcmin,utils.arcmin])*0.1*acc, maxsize=ip_size, maxtime=ip_time, return_obox=True, return_status=True)
	return ipol, obox

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
	nbox[nparam] (grid size along each input parameter), ys[nout,{cval,dx,dy,dz,...},gridsize]."""
	rbox = ipol.box
	nbox = np.array(ipol.ys.shape[4:])
	# ipol.ys has shape [2t,2az,2el,{ra,dec,cos,sin},t,az,el]
	# fortran expects [{ra,dec,cos,sin},{y,dy/dt,dy/daz,dy,del,...},pix]
	# This format allows us to avoid hard-coding the number of input dimensions,
	# and is forward compatible for higher order interpolation later.
	# The disadvantage is that the ordering becomes awkard at higher order.
	n = rbox.shape[1]
	ys = np.asarray([ipol.ys[(0,)*n]] + [ipol.ys[(0,)*i+(1,)+(0,)*(n-i-1)] for i in range(n)])
	ys = np.rollaxis(ys.reshape(ys.shape[:2]+(-1,)),-1).astype(dtype)
	return rbox, nbox, ys

def build_pixbox(obox, n, margin=10):
	res = np.array([np.maximum(0,np.floor(obox[0]-margin)),np.minimum(n,np.floor(obox[1]+margin))]).astype(np.int32)
	# If the original box extends outside [[0,0],n], this box may
	# have empty ranges. If so, return a single-pixel box
	if np.any(res[1]<=res[0]): return np.array([[0,0],[1,1]],dtype=np.int32)
	else: return res

def pmat_phase(dir, tod, map, az, dets, az0, daz):
	core = get_core(tod.dtype)
	core.pmat_phase(dir, tod.T, map.T, az, dets, az0, daz)

def pmat_plain(dir, tod, map, pix):
	core = get_core(tod.dtype)
	core.pmat_plain(dir, tod.T, map.T, pix.T)
