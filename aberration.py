import numpy as np
from enlib import coordinates, enmap as en, utils

beta    = 0.0012301
dir_equ = np.array([167.929,-6.927])*np.pi/180
dir_gal = np.array([263.99, 48.26 ])*np.pi/180

def remap(pos, dir, beta, pol=True, modulation=True, recenter=False):
	"""Given a set of coordinates "pos", computes the aberration
	deflected positions for a speed beta in units of c in the
	direction dir. If pol=True (the default), then the output
	will have three columns, with the third column being
	the aberration-induced rotation of the polarization angle."""
	pos = coordinates.transform("equ",["equ",dir],pos,pol=pol)
	if recenter: before = np.mean(pos[1,::10])
	pos[1] = np.pi/2-aber_angle(np.pi/2-pos[1], beta)
	if recenter:
		after = np.mean(pos[1,::10])
		pos[1] -= after-before
	res = coordinates.transform(["equ",dir],"equ",pos,pol=pol)
	if modulation:
		amp = mod_amplitude(np.pi/2-pos[1], beta)
		res = np.concatenate([res,[amp]])
	return res

def distortion(pos, dir, beta):
	"""Returns the local aberration distortion, defined as the
	second derivative of the aberration displacement."""
	pos = coordinates.transform("equ",["equ",dir],pos,pol=True)
	return aber_deriv(np.pi/2-pos[1], -beta)-1

def aberrate(imap, dir, beta, mode="wrap", order=3, recenter=False):
	pol = imap.ndim > 2
	pos = remap(imap.posmap(), dir, beta, pol=pol, recenter=recenter)
	pix = imap.sky2pix(pos[:2], corner=True) # interpol needs corners
	omap= en.ndmap(utils.interpol(imap, pix, mode=mode, order=order), imap.wcs)
	if pol:
		c,s = np.cos(2*pos[2]), np.sin(2*pos[2])
		omap[1] = c*omap[1] + s*omap[2]
		omap[2] =-s*omap[1] + c*omap[2]
	omap *= pos[3,None]
	return omap

def aber_angle(theta, beta):
	"""The aberrated angle as a function of the input angle."""
	c = np.cos(theta)
	gamma = (1-beta**2)**-0.5
	c = (c+(gamma-1)*c+gamma*beta)/(gamma*(1+c*beta))
	#c = (c+beta)/(1+beta*c)
	return np.arccos(c)

def mod_amplitude(theta, beta):
	c = np.cos(theta)
	gamma = (1-beta**2)**-0.5
	return 1/(gamma*(1-c*beta))
	#return 1/(1-beta*c)

def aber_deriv(theta, beta):
	B = 1-beta**2
	C = 1-beta*np.cos(theta)
	return B**0.5/C
