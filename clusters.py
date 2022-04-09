import numpy as np, pyccl
from pixell import utils, bunch

# Battaglia 12 says:
#
# Δ = 200
#
# Pe  = (2X_H+2)/(5H_H+3) * Pth = 0.5176 Pth
# Pth =
#  P0 * (x/xc)**γ * [(1+(x/xc)**α]**-β' *
#  G M_Δ Δ ρ_cr(z) f_b / (2 R_Δ)
#
# P0(z) = 18.1  * m** 0.154   * (1+z)**-0.758
# xc(z) = 0.497 * m**-0.00865 * (1+z)** 0.731
# β'(z) = 4.35  * m** 0.0393  * (1+z)** 0.415
# α     = 1.00
# γ     = -0.3
#
# m = M_200/(1e14 M_sun)
#
# Relation to standard gnfw parameters:
# β = γ - αβ'
#
# y = k_B sigma_T/(m_e c²) * int dl n_e T_e
#   = sigma_T/(m_e c²) * int dl P_e
#
# Relation of dl and dx, which is what I have in my LOS integral:
# x = l/R_Δ

def get_pkcs_nhalo(fname):
	"""Read rows offset:offset+num of raw data from the given pkcs file.
	if num==0, all values are read"""
	with open(fname, "r") as ifile:
		return np.fromfile(ifile, count=3, dtype=np.uint32)[0]

def read_pkcs_halos(fname, num=0, offset=0):
	"""Read rows offset:offset+num of raw data from the given pkcs file.
	if num==0, all values are read"""
	with open(fname, "r") as ifile:
		n   = np.fromfile(ifile, count=3, dtype=np.uint32)[0]-offset
		if num: n = num
		cat = np.fromfile(ifile, count=n*10, offset=offset*10*4, dtype=np.float32).reshape(n, 10)
		return cat

# This function seems to be just as good as mass_interp, but much simpler
def websky_m200m_to_m200c(m200m, z, cosmology):
	omegam  = cosmology["Omega_m"]
	omegamz = omegam*(1+z)**3/(omegam*(1+z)**3+1-omegam)
	m200c   = omegamz**0.35 * m200m # m200m to m200c conversion used for websky
	return m200c

def decode_websky(data, cosmology, mass_interp):
	"""Go from a raw websky catalog to pos, z and m200"""
	chi     = np.sum(data.T[:3]**2,0)**0.5 # comoving Mpc
	a       = pyccl.scale_factor_of_chi(cosmology, chi)
	z       = 1/a-1
	R       = data.T[6].astype(float) * 1e6*utils.pc # m. This is *not* r200!
	rho_m   = calc_rho_c(cosmology, 0)*cosmology["Omega_m"]
	m200m   = 4/3*np.pi*rho_m*R**3
	m200    = mass_interp(m200m, z)
	ra, dec = utils.rect2ang(data.T[:3])
	return bunch.Bunch(z=z, ra=ra, dec=dec, m200=m200)

def get_halo_params(m200, z, cosmology):
	"""Compute gnfw parameters and the physical normalization based on m200 and redshift"""
	# Position, redshift and size
	r200    = calc_rdelta (m200, z, cosmology)
	rang    = calc_angsize(r200, z, cosmology)
	rho_c   = calc_rho_c(cosmology, z)
	# GNFW parameters given this
	params= get_params_battaglia(m200, z, cosmology)
	# Compute the factor that takes us from what tsz_profile_los()
	# returns to the compton y parameter
	# Pe  = (2X_H+2)/(5H_H+3) * Pth = 0.5176 Pth
	# Pth = G M_Δ Δ ρ_cr(z) f_b / (2 R_Δ) * P0 * gnfw()
	# y   = sigma_T/(m_e c²) * int dl P_e
	#     = sigma_T/(m_e c²) * 0.5176 * G M_Δ Δ ρ_cr(z) f_b / (2 R_Δ) * R_Δ * P0 * tsz_profile_los()
	# Here the extra R_Δ comes from the radial integral, dl = R_Δ dx
	sigma_T = 6.6524587158e-29 # m²
	m_e     = 9.10938356e-31   # kg
	f_b     = cosmology["Omega_b"]/cosmology["Omega_m"]
	y_conv  = sigma_T/(m_e*utils.c**2) * 0.5176 * utils.G * m200 * 200 * rho_c * f_b / (2*r200) * r200
	#print(0.5176 * utils.G * m200 * 200 * rho_c * f_b / (2*r200) * params.P0)
	# Unit m² / (kg*m²/s²) * 1/kg m³/s² * kg * kg/m³ = 1 OK
	return bunch.Bunch(r200=r200, rang=rang, y_conv=y_conv, **params)

def get_H0(cosmology): return cosmology["h"]*100*1e3/(1e6*utils.pc)
def calc_rho_c(cosmology, z):
	H     = get_H0(cosmology)*pyccl.h_over_h0(cosmology, 1/(z+1))
	rho_c = 3*H**2/(8*np.pi*utils.G)
	return rho_c

def calc_rdelta(mdelta, z, cosmology, delta=200):
	"""Given M_delta in kg, returns R_delta in m"""
	# Why do I have to access a private member to get H0?
	rho_c  = calc_rho_c(cosmology, z)
	rdelta = (mdelta/(4/3*np.pi*delta*rho_c))**(1/3)
	return rdelta

def calc_angsize(physsize, z, cosmology):
	"""Given a physical size in m, returns the angular size in radians"""
	d_A = pyccl.angular_diameter_distance(cosmology, 1/(z+1))*1e6*utils.pc
	return np.arctan2(physsize,d_A)

def get_params_battaglia(m200, z, cosmology):
	"""Return a bunch of xc, alpha, beta, gamma for a cluster with
	the given m200 in SI units."""
	# First get the gnfw parameters. utils.gnfw has the opposite sign for
	# beta and gamma as nemo, but otherwise the same convention
	z1    = z+1
	m     = m200/(1e14*utils.M_sun)
	P0    =  18.1     * m** 0.154   * z1**-0.758
	xc    =  0.497    * m**-0.00865 * z1** 0.731
	beta  =  4.35     * m** 0.0393  * z1** 0.415
	alpha =  1; gamma = -0.3
	# Go from battaglia convention to standard gnfw
	beta  = gamma - alpha*beta
	return bunch.Bunch(xc=xc, alpha=alpha, beta=beta, gamma=gamma, P0=P0)

def eval_battaglia_slow(mdelta, zs, rang, cosmology, rs):
	rprofs = []
	for i, (mdelta, z, rang) in enumerate(zip(mdeltas, zs, rangs)):
		params = get_params_battaglia(mdelta, z, cosmology)
		# 29 ms per object. Slow... cache={} to prevent caching of single-use params
		rprofs.append(params.P0*utils.tsz_profile_los(rs/rang, xc=params.xc, alpha=params.alpha, beta=params.beta, gamma=params.gamma, cache={}))
	return np.array(rprofs)

class FastBattagliaInterp:
	def __init__(self, beta1, beta2, cosmology, alpha=1, gamma=-0.3,
			obeta=4, nbeta=None, npoint=200, pad=0.1, x1=1e-10, x2=1e8, _a=8, zmax=1e5):
		# I first thought an xc,beta expansion would be necessary here, but
		# xc just enters as a scaling of x anyway, and we handle x interpolation.
		# NB: xc also scales the dz integral in the LOS calculation
		# If the number of beta samples are not specified, use one point per basis function
		if nbeta is None: nbeta = obeta
		# The radial values to evlauate things at
		xp = np.linspace(np.log(x1),np.log(x2),npoint)
		# typically beta is */ 1.5. After log that's ± 0.4
		lbeta1, lbeta2 = np.log([-beta2, -beta1])
		lbeta0 = 0.5*(lbeta2+lbeta1)
		dlbeta = 0.5*(lbeta2-lbeta1)+pad
		lbetas = np.linspace(lbeta0-dlbeta, lbeta0+dlbeta, nbeta)
		# Evaluate for all combinations
		yps = np.array([utils.tsz_profile_los_exact(np.exp(xp), xc=1, alpha=alpha, beta=-np.exp(lbeta), gamma=gamma, zmax=zmax, _a=_a) for lbeta in lbetas]) # [nbeta,npoint]
		yps = np.log(yps)
		# I hoped scipy would have an efficient 2d interpolator, but I could only find
		# map_coordinates and RegularGridInterpolator. The former has bad boundary conditions,
		# and the latter is just a subset of the former. I'll just do some polynomial
		# interpolation myself.
		B = ((lbetas-lbeta0)/dlbeta)**(np.arange(obeta)[:,None]) # [obeta,nbeta]
		# Fit our data points to this basis
		a = np.linalg.solve(B.dot(B.T), B.dot(yps)) # [obeta,npoint]
		# Prefilter a to make utils.interpol faster
		a = utils.interpol_prefilter(a, npre=1, inplace=True)
		# We have everything we need for interpolation now
		self.lbeta0 = lbeta0
		self.dlbeta = dlbeta
		self.xp0    = xp[0]
		self.dxp    = xp[1]-xp[0]
		self.B  = B
		self.a  = a
		self.npoint = npoint
		self.obeta  = obeta
	def __call__(self, betas, xcs, P0s, rangs, r):
		"""betas: betas to evaluate profiles for. [ncase]
		xs = r/rang/xc. Scaled values to evaluate profiles as [ncase,nx]"""
		# Broadcast and reshape to the expected shapes
		betas, xcs, P0s, rangs = np.broadcast_arrays(betas, xcs, P0s, rangs)
		r      = np.array(r)
		bshape = betas.shape
		rshape = r.shape
		betas, xcs, P0s, rangs = [a.reshape(-1) for a in [betas, xcs, P0s, rangs]]
		r      = r.reshape(-1)
		# Nomralize beta so it matches what our polynomial basis expects
		b    = (np.log(-np.array(betas))-self.lbeta0)/self.dlbeta
		B    = b**(np.arange(self.obeta)[:,None]) # [obeta,ncase]
		# Get our dimensionless x values. Absorb xc into these since
		# we've built our interpolation around xc=1, but remember to compensate
		# for the unit change this implies before returning
		x     = r/(rangs*xcs)[:,None]
		x     = np.maximum(x, 1e-10)
		xpix = (np.log(x)-self.xp0)/self.dxp
		ncase, nx = xpix.shape
		# This is inefficient if we have more interp points than output points
		ainter = utils.interpol(self.a, xpix.T[None], prefilter=False) # [obeta,nx,ncase]
		profs  = np.sum(B[:,None,:]*ainter,0).T
		profs  = np.exp(profs)*xcs[:,None]*P0s[:,None]
		# Reshape result to match inputs
		profs  = profs.reshape(bshape+rshape)
		return profs

class MdeltaTranslator:
	def __init__(self, cosmology,
			type1="matter", delta1=200, type2="critical", delta2=200,
			zlim=[0,20], mlim=[1e11*utils.M_sun,5e16*utils.M_sun], step=0.1):
		idef = pyccl.halos.MassDef(delta1, type1, c_m_relation="Bhattacharya13")
		odef = pyccl.halos.MassDef(delta2, type2, c_m_relation="Bhattacharya13")
		# Set up our sample grid, which will be log-spaced in both z and mass direction
		lz1, lz2 = np.log(1+np.array(zlim)) # lz = log(1+z) = -log(a)
		lm1, lm2 = np.log(np.array(mlim))   # lm = log(m)
		nz  = utils.ceil((lz2-lz1)/step)
		nm  = utils.ceil((lm2-lm1)/step)
		lzs = np.linspace(lz1, lz2, nz)
		lms = np.linspace(lm1, lm2, nm)
		olms = np.zeros((len(lzs),len(lms)))
		for ai, lz in enumerate(lzs):
			moo = np.exp(lms[-1])/utils.M_sun
			olms[ai] = idef.translate_mass(cosmology, np.exp(lms)/utils.M_sun, np.exp(-lz), odef)
		olms = np.log(olms*utils.M_sun)
		olms = utils.interpol_prefilter(olms, order=3)
		# Save parameters
		self.lz1, self.lz2, self.dlz = lz1, lz2, (lz2-lz1)/(nz-1)
		self.lm1, self.lm2, self.dlm = lm1, lm2, (lm2-lm1)/(nm-1)
		self.olms = olms
	def __call__(self, m, z):
		zpix = (np.log(1+np.array(z))-self.lz1)/self.dlz
		mpix = (np.log(m)-self.lm1)/self.dlm
		pix  = np.array([zpix,mpix])
		return np.exp(utils.interpol(self.olms, pix, order=3, prefilter=False))
