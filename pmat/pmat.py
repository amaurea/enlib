"""This module contains implementations for "pointing matrices", which project
between time-ordered data (described by a Scan class) down to some other space,
such as a map. Its general interface is .forward(tod,m) and .backward(tod,m).
Both of these are allowed to modify both arguments, though usually
forward will update tod based on m, and backward will update m based on tod.
The reason for allowing the other argument to be modified is to make it easier
to incrementally project different parts of the signal.
"""
import numpy as np, bunch
from enlib import enmap, interpol, utils, coordinates, config, errors
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
config.default("pmat_accuracy",     1.0, "Factor by which to lower accuracy requirement in pointing interpolation. 1.0 corresponds to 1e-2 pixels and 1 arc minute in polangle")
config.default("pmat_interpol_max_size", 100000, "Maximum mesh size in pointing interpolation. Worst-case time and memory scale at most proportionally with this.")
config.default("pmat_interpol_max_time", 50, "Maximum time to spend in pointing interpolation constructor. Actual time spent may be up to twice this.")

class PointingMatrix:
	def forward(self, tod, m): raise NotImplementedError
	def backward(self, tod, m): raise NotImplementedError

class PmatMap(PointingMatrix):
	"""Fortran-accelerated scan <-> enmap pointing matrix implementation.
	20 times faster than the slower python+numpy implementation below."""
	def __init__(self, scan, template, sys=None, order=None):
		sys   = config.get("map_eqsys",      sys)
		order = config.get("pmat_map_order", order)

		box = np.array(scan.box)
		margin = (box[1]-box[0])*1e-3 # margin to avoid rounding erros
		box[0] -= margin/2; box[1] += margin/2
		acc  = config.get("pmat_accuracy")
		ip_size= config.get("pmat_interpol_max_size")
		ip_time= config.get("pmat_interpol_max_time")
		transform = pos2pix(scan,template,sys)
		ipol = interpol.build(transform, interpol.ip_linear, box, np.array([1e-2,1e-2,utils.arcmin,utils.arcmin])*acc, maxsize=ip_size, maxtime=ip_time)
		self.rbox = ipol.box
		self.nbox = np.array(ipol.ys.shape[4:])
		# ipol.ys has shape [2t,2az,2el,{ra,dec,cos,sin},t,az,el]
		# fortran expects [{ra,dec,cos,sin},{y,dy/dt,dy/daz,dy,del,...},pix]
		# This format allows us to avoid hard-coding the number of input dimensions,
		# and is forward compatible for higher order interpolation later.
		# The disadvantage is that the ordering becomes awkard at higher order.
		n = self.rbox.shape[1]
		self.ys = np.asarray([ipol.ys[(0,)*n]] + [ipol.ys[(0,)*i+(1,)+(0,)*(n-i-1)] for i in range(n)])
		self.ys = np.rollaxis(self.ys.reshape(self.ys.shape[:2]+(-1,)),-1).astype(template.dtype)
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
	def forward(self, tod, m):
		"""m -> tod"""
		self.func( 1, tod.T, m.T, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T)
	def backward(self, tod, m):
		"""tod -> m"""
		self.func(-1, tod.T, m.T, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T)
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
		return ({"none":0,"full":1,"bin":2,"exp":3}[toks[0]],)+args

class pos2pix:
	"""Transforms from scan coordintaes to pixel-center coordinates."""
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
			opix[:2] = opos[1::-1]
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

config.default("pmat_ptsrc_rsigma", 5.0, "Max number of standard deviations away from a point source to compute the beam profile. Larger values are slower but more accurate.")
class PmatPtsrc(PointingMatrix):
	def __init__(self, scan, params, sys=None, tmul=None, pmul=None):
		sys   = config.get("map_eqsys", sys)
		rmul  = config.get("pmat_ptsrc_rsigma")
		self.dtype = params.dtype

		# Set up pointing interpolation
		box = np.array(scan.box)
		margin = (box[1]-box[0])*1e-3 # margin to avoid rounding erros
		box[0] -= margin/2; box[1] += margin/2
		ref_phi = params[1,0]
		acc  = config.get("pmat_accuracy")
		ip_size= config.get("pmat_interpol_max_size")
		ip_time= config.get("pmat_interpol_max_time")
		transform = pos2pix(scan,None,sys,ref_phi=ref_phi)
		ipol = interpol.build(transform, interpol.ip_linear, box, np.array([utils.arcsec, utils.arcsec ,utils.arcmin,utils.arcmin])*acc, maxsize=ip_size, maxtime=ip_time)
		self.rbox = ipol.box
		self.nbox = np.array(ipol.ys.shape[4:])
		n = self.rbox.shape[1]
		self.ys = np.asarray([ipol.ys[(0,)*n]] + [ipol.ys[(0,)*i+(1,)+(0,)*(n-i-1)] for i in range(n)])
		self.ys = np.rollaxis(self.ys.reshape(self.ys.shape[:2]+(-1,)),-1).astype(self.dtype)
		self.comps = np.arange(params.shape[0]-5)
		self.scan  = scan
		self.core = pmat_core_32.pmat_core if self.dtype == np.float32 else pmat_core_64.pmat_core

		# Collect information about which samples hit which sources
		nsrc = params.shape[1]
		ndet = scan.ndet
		# rhit is the inverse of the squared amplitude-weighted inverse beam for
		# some reason. But it is basically going to be our beam size.
		rhit = np.zeros(nsrc)+(np.sum(1./params[-3]*params[2]**2)/np.sum(params[2]**2))**0.5
		rmax = rhit*rmul
		src_ivars = np.zeros(nsrc,dtype=self.dtype)

		# Measure ranges. May need to iterate if initial allocation was too small
		nrange = np.zeros([nsrc,ndet],dtype=np.int32)
		ranges = np.zeros([nsrc,ndet,100,2],dtype=np.int32)
		self.core.pmat_ptsrc_prepare(params, rhit, rmax, scan.noise.ivar, src_ivars, ranges.T, nrange.T, self.scan.boresight.T, self.scan.offsets.T, self.rbox.T, self.nbox, self.ys.T)
		if np.max(nrange) > ranges.shape[2]:
			ranges = np.zeros([nsrc,ndet,np.max(nrange),2],dtype=np.int32)
			self.core.pmat_ptsrc_prepare(params, rhit, rmax, scan.noise.ivar, src_ivars, ranges.T, nrange.T, self.scan.boresight.T, self.scan.offsets.T, self.rbox.T, self.nbox, self.ys.T)

		self.ranges, self.rangesets, self.offsets = compress_ranges(ranges, nrange, scan.cut, scan.nsamp)
		self.src_ivars = src_ivars
		self.nhit = np.sum(self.ranges[:,1]-self.ranges[:,0])

		self.tmul = 0 if tmul is None else tmul
		self.pmul = 1 if pmul is None else pmul

	def forward(self, tod, params, tmul=None, pmul=None):
		"""params -> tod"""
		if tmul is None: tmul = self.tmul
		if pmul is None: pmul = self.pmul
		self.core.pmat_ptsrc(tmul, pmul, tod.T, params, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T, self.ranges.T, self.rangesets, self.offsets.T)

	def extract(self, tod):
		"""Extract precomputed pointing and phase information for the selected samples.
		These are stored in a compressed format, where sample I of range R of detector D
		of source S has index ranges[offsets[S,D]+R,0]+I.

		Ideally the ranges should cover the data once and no more, to avoid double-counting
		and unsubtracted sources when several sources are near each other. In that case,
		overlapping ranges must be merged, and both sources must be made to refer to the
		same range, meaning that there will be duplicate entries in offsets."""
		point = np.zeros([self.nhit,2],dtype=self.dtype)
		phase = np.zeros([self.nhit,len(self.comps)],dtype=self.dtype)
		srctod= np.zeros([self.nhit],dtype=self.dtype)
		oranges= np.zeros(self.ranges.shape, dtype=np.int32)
		self.core.pmat_ptsrc_extract(tod.T, srctod, point.T, phase.T, oranges.T, self.scan.boresight.T,
				self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox,
				self.ys.T, self.ranges.T, self.rangesets, self.offsets.T)
		res = bunch.Bunch(point=point, phase=phase, tod=srctod, ranges=oranges, rangesets=self.rangesets, offsets=self.offsets, dets=self.scan.dets)
		return res

def compress_ranges(ranges, nrange, cut, nsamp):
	"""Given ranges[nsrc,ndet,nmax,2], nrange[nsrc,ndet] where ranges has
	det-local numbering, return the same information in a compressed format
	ranges[nr,2], rangesets[nind], offsets[nsrc,ndet], where ranges now has
	global sample ordering."""
	nsrc, ndet = nrange.shape
	# Special case: None hit. We represent this as a single range hitting no samples,
	# which isn't used by any of the srcs.
	if np.sum(nrange) == 0:
		ranges  = np.array([[0,0]],dtype=np.int32)
		rangesets = np.array([0],dtype=np.int32)
		offsets = np.zeros([nsrc,ndet+1],dtype=np.int32)
		return ranges, rangesets, offsets
	# First collapse ranges,nrange to flat ranges and indices into it
	flat_ranges = []
	offsets = np.zeros([nsrc,ndet+1],dtype=np.int32)
	for si in range(nsrc):
		for di in range(ndet):
			s0 = di*nsamp
			offsets[si,di] = len(flat_ranges)
			current_ranges = ranges[si,di,:nrange[si,di]]
			cutsplit_ranges = utils.range_sub(current_ranges, cut[di].ranges)
			flat_ranges += list(cutsplit_ranges+s0)
			offsets[si,di+1] = len(flat_ranges)
	flat_ranges = np.atleast_2d(flat_ranges)
	# Then merge overlapping ranges and produce our final output format.
	ranges, map = utils.range_union(flat_ranges, mapping=True)
	# Because some ranges are shared now, we need a list of which
	# range indices belongs to which src,det. That is what map
	# does. offsets, which used to be indices into ranges, are now
	# indices into map instead.
	return ranges, map, offsets
