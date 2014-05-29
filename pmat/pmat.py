"""This module contains implementations for "pointing matrices", which project
between time-ordered data (described by a Scan class) down to some other space,
such as a map. Its general interface is .forward(tod,m) and .backward(tod,m).
Both of these are allowed to modify both arguments, though usually
forward will update tod based on m, and backward will update m based on tod.
The reason for allowing the other argument to be modified is to make it easier
to incrementally project different parts of the signal.

We need to be very clear about how we interpret pixel coordinates here.
We assign the value of a pixel to its center point. In nearest neighbor
interpolation this is also the hit-average inside that pixel. This means
that the integer pixel index to assign to will be ipix = round(pix),
not floor(pix). Does it?

     0    1    2    3
  +----+----+----+----+

If the floating point pixel coordinates are relative to pixel
centers, then we should use round. Otherwise it's still floor.
"""
import numpy as np
from enlib import enmap, interpol, utils, coordinates
from pmat_core import pmat_core

class PointingMatrix:
	def forward(self, tod, m): raise NotImplementedError
	def backward(self, tod, m): raise NotImplementedError

class pos2pix:
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

class MapPointingMatrixNearestSlow(PointingMatrix):
	"""Reference implementation of the simple nearest neighbor
	pointing matrix. Slow - not meant for serious use. It's interesting
	as an example of how a reasonable python-only implementation would
	perform."""
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
			print np.min(pix,1), np.max(pix,1)
			for c in range(m.shape[0]):
				m[c] += np.bincount(np.ravel_multi_index(pix[:2], m.shape[-2:]), tod[d], m[c].size).reshape(m[c].shape)
			print np.min(m), np.max(m)
	def forward(self, tod, m):
		for d in range(tod.shape[0]):
			print d
			pix, phase = self.get_pix_phase(d)
			tod[d] += np.sum(m[:,pix[0],pix[1]]*phase[:m.shape[0]],0)

# To go faster, we need two sets of interpolation coefficients in fotran.
# The first is the one used to compute the pixel coordinates of each sample.
# This can be the linear coefficients from ip_linear. ip_linear itself is
# relatively slow, but it only needs to be called when building the pointing
# matrix, and takes a few sec. We then use these in a much faster fortran
# implementation.
#
# The second level of interpolation is the pixel value interpolation. These
# interpolation coefficients are too many to be precomputed. They must be
# computed on the fly in fortran.

class MapPointingMatrixNearestFast(PointingMatrix):
	def __init__(self, scan, template, sys="equ"):
		box = np.array(scan.box); box[1] += (box[1]-box[0])*1e-3 # margin to avoid rounding errors
		ipol = interpol.build(pos2pix(scan,template,sys), interpol.ip_linear, box, [1e-3,1e-3,utils.arcsec,utils.arcsec])
		self.rbox = ipol.box
		self.nbox = np.array(ipol.ys.shape[4:])
		# ipol.ys has shape [2t,2az,2el,{ra,dec,cos,sin},t,az,el]
		# fortran expects [{ra,dec,cos,sin},2el,2az,2t,pix]
		# so flatten into pix, and move pix first
		# ys is flattened in C order.
		# Since I will almost always want just y and grad y, let's order ys as
		# [y,dy/dt,dy/daz,dy/del,...]. That way the lower derivatives don't need
		# to worry about the presence of higher ones
		n = self.rbox.shape[1]
		self.ys = np.asarray([ipol.ys[(0,)*n]] + [ipol.ys[(0,)*i+(1,)+(0,)*(n-i-1)] for i in range(n)])
		self.ys = np.rollaxis(self.ys.reshape(self.ys.shape[:2]+(-1,)),-1).astype(np.float32)
		self.comps= np.arange(template.shape[0])
		self.scan = scan
	def forward(self, tod, m):
		pmat_core.map_nearest( 1, tod.T, m.T, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T)
	def backward(self, tod, m):
		pmat_core.map_nearest(-1, tod.T, m.T, self.scan.boresight.T, self.scan.offsets.T, self.scan.comps.T, self.comps, self.rbox.T, self.nbox, self.ys.T)
