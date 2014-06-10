"""This module contains implementations for "pointing matrices", which project
between time-ordered data (described by a Scan class) down to some other space,
such as a map. Its general interface is .forward(tod,m) and .backward(tod,m).
Both of these are allowed to modify both arguments, though usually
forward will update tod based on m, and backward will update m based on tod.
The reason for allowing the other argument to be modified is to make it easier
to incrementally project different parts of the signal.
"""
import numpy as np
from enlib import enmap, interpol, utils, coordinates, config
from pmat_core import pmat_core

config.default("pmat_map_order",      0, "The interpolation order of the map pointing matrix.")
config.default("pmat_cut_type",  "full", "The cut sample representation used. 'full' uses one degree of freedom for each cut sample. 'bin:N' uses one degree of freedom for every N samples. 'exp' used one degree of freedom for the first sample, then one for the next two, one for the next 4, and so on, giving high resoultion at the edges of each cut range, and low resolution in the middle.")
config.default("map_eqsys",       "equ", "The coordinate system of the maps. Can be eg. 'hor', 'equ' or 'gal'.")

class PointingMatrix:
	def forward(self, tod, m): raise NotImplementedError
	def backward(self, tod, m): raise NotImplementedError

class PmatMap(PointingMatrix):
	"""Fortran-accelerated scan <-> enmap pointing matrix implementation.
	20 times faster than the slower python+numpy implementation below."""
	def __init__(self, scan, template, sys=None, order=None):
		sys   = config.get("map_eqsys",      sys)
		order = config.get("pmat_map_order", order)

		box = np.array(scan.box); box[1] += (box[1]-box[0])*1e-3 # margin to avoid rounding errors
		ipol = interpol.build(pos2pix(scan,template,sys), interpol.ip_linear, box, [1e-3,1e-3,utils.arcsec,utils.arcsec])
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
		if order == 0:
			self.func = pmat_core.pmat_nearest
		elif order == 1:
			self.func = pmat_core.pmat_linear
		else:
			raise NotImplementedError("order > 1 is not implemented")
	def forward(self, tod, m):
		self.func( 1, tod.T, m.T, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T)
	def backward(self, tod, m):
		self.func(-1, tod.T, m.T, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T)

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
		neach, flat = scan.cut.flatten()
		dets = np.concatenate([np.zeros(n,dtype=int)+i for i,n in enumerate(neach)])
		par  = np.array(self.parse_params(params))
		self.cuts = np.zeros([flat.shape[0],5+len(par)],dtype=np.int32)
		self.cuts[:,0] = dets
		self.cuts[:,1] = flat[:,0]
		self.cuts[:,2] = flat[:,1]-flat[:,0]
		self.cuts[:,5:]= par[None,:]
		if self.cuts.size > 0:
			pmat_core.measure_cuts(self.cuts.T)
		self.cuts[:,3] = utils.cumsum(self.cuts[:,4])
		# njunk is the number of cut parameters for *this scan*
		self.njunk  = np.sum(self.cuts[:,4])
		self.params = params
		self.scan = scan
	def forward(self, tod, junk):
		"""Project from the cut parameter (junk) space for this scan
		to tod."""
		if self.cuts.size > 0:
			pmat_core.pmat_cut( 1, tod.T, junk, self.cuts.T)
	def backward(self, tod, junk):
		"""Project from tod to cut parameters (junk) for this scan.
		This is meant to be called before the map projection, and
		removes the cut samples from the tod at the same time,
		replacing them with zeros. That way the map projection can
		be done without needing to care about the cuts."""
		if self.cuts.size > 0:
			pmat_core.pmat_cut(-1, tod.T, junk, self.cuts.T)
	def parse_params(self,params):
		toks = params.split(":")
		kind = toks[0]
		args = tuple([int(s) for s in toks[1].split(",")]) if len(toks) > 1 else ()
		return ({"none":0,"full":1,"bin":2,"exp":3}[toks[0]],)+args

class pos2pix:
	"""Transforms from scan coordintaes to pixel-center coordinates."""
	def __init__(self, scan, template, sys):
		self.scan, self.template, self.sys = scan, template, sys
	def __call__(self, ipos):
		shape = ipos.shape[1:]
		ipos  = ipos.reshape(ipos.shape[0],-1)
		time  = self.scan.mjd0 + ipos[0]/utils.day2sec
		opos = coordinates.transform(self.scan.sys, self.sys, ipos[1:], time=time, site=self.scan.site, pol=True)
		opix = np.zeros((4,)+ipos.shape[1:])
		opix[:2] = self.template.sky2pix(opos[1::-1])
		opix[2]  = np.cos(2*opos[2])
		opix[3]  = np.sin(2*opos[2])
		return opix.reshape((opix.shape[0],)+shape)
