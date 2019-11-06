import cython, numpy as np
from .. import enmap, utils
from scipy import ndimage

__version__ = 1.0

cdef extern from "parallax.h":
	void displace_map_blocks_avx_omp(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx)
	void solve_plain(float * frhs, float * kmap, float * osigma, int ny, int nx, float klim)
	void update_total_plain(float * sigma, float * sigma_max, float * param_max, int * hit_tot, float * kmap, int ny, int nx, float r, float vy, float vx)
	#void displace_map2(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx)
	#void displace_map3(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx)
	#void displace_map4(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx)
	#void displace_map5(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx)
	#void displace_map6(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx)
	#void displace_map7(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx)
	#void displace_map8(float * imap, float * omap, int ny, int nx, double dec0, double ddec, double ra0, double dra, double * earth_pos, double r, double dy, double dx)

def displace_map(imap, earth_pos, r, off, omap=None, method=1):
	# Check that we have the right input data types
	assert imap.ndim == 2, "imap must be 2D"
	assert imap.flags["C_CONTIGUOUS"], "imap must be contiguous"
	assert imap.dtype == np.float32, "imap must be float"
	if method == 0: return displace_map0(imap, earth_pos, r, off, omap)
	if omap is None: omap = enmap.zeros(imap.shape, imap.wcs, imap.dtype)
	# Extract the pixelization
	cdef double ra0  = utils.degree*(imap.wcs.wcs.crval[0] - imap.wcs.wcs.cdelt[0]*imap.wcs.wcs.crpix[0])
	cdef double dec0 = utils.degree*(imap.wcs.wcs.crval[1] - imap.wcs.wcs.cdelt[1]*imap.wcs.wcs.crpix[1])
	cdef double dra  = utils.degree*(imap.wcs.wcs.cdelt[0])
	cdef double ddec = utils.degree*(imap.wcs.wcs.cdelt[1])
	cdef int ny = imap.shape[0]
	cdef int nx = imap.shape[1]
	# Make memory views we can pass to C
	cdef double[::1] earth_pos_ = np.array(earth_pos, np.float64)
	cdef float[::1] imap_      = imap.reshape(-1)
	cdef float[::1] omap_      = omap.reshape(-1)
	if method == 1:
		displace_map_blocks_avx_omp(&imap_[0], &omap_[0], ny, nx, dec0, ddec, ra0, dra, &earth_pos_[0], r, off[0], off[1])
	#elif method == 2:
	#	displace_map2(&imap_[0], &omap_[0], ny, nx, dec0, ddec, ra0, dra, &earth_pos_[0], r, off[0], off[1])
	#elif method == 3:
	#	displace_map3(&imap_[0], &omap_[0], ny, nx, dec0, ddec, ra0, dra, &earth_pos_[0], r, off[0], off[1])
	#elif method == 4:
	#	displace_map4(&imap_[0], &omap_[0], ny, nx, dec0, ddec, ra0, dra, &earth_pos_[0], r, off[0], off[1])
	#elif method == 5:
	#	displace_map5(&imap_[0], &omap_[0], ny, nx, dec0, ddec, ra0, dra, &earth_pos_[0], r, off[0], off[1])
	#elif method == 6:
	#	displace_map6(&imap_[0], &omap_[0], ny, nx, dec0, ddec, ra0, dra, &earth_pos_[0], r, off[0], off[1])
	#elif method == 7:
	#	displace_map7(&imap_[0], &omap_[0], ny, nx, dec0, ddec, ra0, dra, &earth_pos_[0], r, off[0], off[1])
	#elif method == 8:
	#	displace_map8(&imap_[0], &omap_[0], ny, nx, dec0, ddec, ra0, dra, &earth_pos_[0], r, off[0], off[1])
	else: raise ValueError("Unknown method %d" % method)
	return omap

# Direct python implementation for comparison
def displace_map0(imap, earth_pos, r, off, omap=None):
	if omap is None: omap = enmap.zeros(imap.shape, imap.wcs, imap.dtype)
	posmap = omap.posmap()
	cdec   = np.cos(posmap[0])
	posmap[0] += off[0]
	posmap[1] += off[1]/cdec
	rect   = utils.ang2rect(posmap)*r
	rect  += np.array(earth_pos)[:,None,None]
	ipos   = utils.rect2ang(rect)
	ipix   = imap.sky2pix(ipos)
	omap[:] = ndimage.map_coordinates(imap, ipix, order=1)
	return omap

def solve(frhs, kmap, osigma=None, klim=0):
	if osigma is None: osigma = enmap.empty(frhs.shape, frhs.wcs, frhs.dtype)
	# Make memory views
	cdef float[::1] frhs_ = frhs.reshape(-1)
	cdef float[::1] kmap_ = kmap.reshape(-1)
	cdef float[::1] osigma_ = osigma.reshape(-1)
	solve_plain(&frhs_[0], &kmap_[0], &osigma_[0], frhs.shape[0], frhs.shape[1], klim)
	return osigma

def update_total(sigma, sigma_max, param_max, hit_tot, kmap, r, vy, vx):
	cdef float[::1] sigma_    = sigma.reshape(-1)
	cdef float[::1] sigma_max_ = sigma_max.reshape(-1)
	cdef float[::1] param_max_ = param_max.reshape(-1)
	cdef float[::1] kmap_      = kmap.reshape(-1)
	cdef int[::1]   hit_tot_ = hit_tot.reshape(-1)
	update_total_plain(&sigma_[0], &sigma_max_[0], &param_max_[0], &hit_tot_[0], &kmap_[0], sigma.shape[0], sigma.shape[1], r, vy, vx)
