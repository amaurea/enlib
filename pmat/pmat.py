"""This module contains implementations for "pointing matrices", which project
between time-ordered data (described by a Scan class) down to some other space,
such as a map. Its general interface is .forward(tod,m) and .backward(tod,m).
Both of these are allowed to modify both arguments, though usually
forward will update tod based on m, and backward will update m based on tod.
The reason for allowing the other argument to be modified is to make it easier
to incrementally project different parts of the signal.
"""
import numpy as np
from enlib import enmap, interpol, utils, coordinates
from pmat_core import pmat_core

class PointingMatrix:
	def forward(self, tod, m): raise NotImplementedError
	def backward(self, tod, m): raise NotImplementedError

class MapPmat(PointingMatrix):
	"""Fortran-accelerated scan <-> enmap pointing matrix implementation.
	20 times faster than the slower python+numpy implementation below."""
	def __init__(self, scan, template, sys="equ", order=0):
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
		self.ys = np.rollaxis(self.ys.reshape(self.ys.shape[:2]+(-1,)),-1).astype(np.float32)
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

class MapPmatSlow(PointingMatrix):
	"""Reference implementation of the simple nearest neighbor
	pointing matrix. Very slow - not meant for serious use. It's interesting
	as an example of how a reasonable python-only implementation would
	perform. Uses trilinear interpolation for coordinates."""
	def __init__(self, scan, template, sys="equ"):
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
