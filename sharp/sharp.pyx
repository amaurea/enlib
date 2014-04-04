import numpy as np
cimport numpy as np
cimport csharp
from libc.math cimport atan2

cdef class map_info:
	"""This class is a thin wrapper for the sharp geom_info struct, which represents
	the pixelization used in spherical harmonics transforms. It can represent an
	abritrary number of constant-latitude rings at arbitrary latitudes. Each ring
	has an arbitrary number of equidistant points in latitude, with an abritrary
	offset in latitude."""
	cdef csharp.sharp_geom_info * geom
	cdef readonly int nrow
	cdef readonly int npix
	cdef readonly np.ndarray theta
	cdef readonly np.ndarray nphi
	cdef readonly np.ndarray phi0
	cdef readonly np.ndarray offsets
	cdef readonly np.ndarray stride
	cdef readonly np.ndarray weight
	def __cinit__(self, theta, nphi=0, phi0=0, offsets=None, stride=None, weight=None):
		"""Construct a new sharp map geometry consiting of N rings in co-latitude, at
		co-latitudes theta[N], with the rings having nphi[N] points each, with the
		first point at longitude phi0[N]. Each ring has pixel stride stride[N], and
		the pixel index of the first pixel in each ring is offsets[N]. For map2alm
		transforms, weight[N] specifies the integral weights for each row. These
		are complicated and depend on the pixel layout."""
		theta = np.asarray(theta, dtype=np.float64)
		assert(theta.ndim == 1, "theta must be one-dimensional!")
		ntheta = len(theta)
		nphi  = np.asarray(nphi, dtype=np.int32)
		assert(nphi.ndim < 2, "nphi must be 0 or 1-dimensional")
		if nphi.ndim == 0:
			nphi = np.zeros(ntheta,dtype=np.int32)+(nphi or 2*ntheta)
		assert(len(nphi) == ntheta, "theta and nphi arrays do not agree on number of rings")
		phi0 = np.asarray(phi0, dtype=np.float64)
		assert(phi0.ndim < 0, "phi0 must be 0 or 1-dimensional")
		if phi0.ndim == 0:
			phi0 = np.zeros(ntheta,dtype=np.float64)+phi0
		if offsets is None:
			offsets = np.concatenate([[0],np.cumsum(nphi.astype(np.int64)[:-1])])
		if stride  is None:
			stride  = np.zeros(ntheta,dtype=np.int32)+1
		if weight  is None:
			weight  = np.zeros(ntheta,dtype=np.float64)+1
		self.geom = make_geometry_helper(ntheta, nphi, offsets, stride, phi0, theta, weight)
		# Store publicly accessible view of the internal geometry. This
		# is kept by going out of sync via readonly and writable=False
		self.npix = np.sum(nphi)
		self.nrow = len(nphi)
		for v in [theta,nphi,phi0,offsets,stride,weight]: v.flags.writeable = False
		self.theta, self.nphi, self.phi0, self.offsets, self.stride, self.weight = theta, nphi, phi0, offsets, stride, weight
	def __dealloc__(self):
		csharp.sharp_destroy_geom_info(self.geom)

def map_info_healpix(int nside, int stride=1, weights=None):
	"""Construct a new sharp map geometry for the HEALPix pixelization in the RING
	scheme, with the given nside parameter (which does not need to be a power of
	2 in this case). The optional weights array specifies quadrature weights
	*relative* to the default weights, so you will get sensible results even without
	specifying weights."""
	nring = 4*nside-1
	if weights is None: weights = np.zeros(nring)+1
	assert(len(weights) < nring, "incorrect length of weights array. need 4*nside-1")
	cdef np.ndarray[np.float64_t,ndim=1] w = weights
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_weighted_healpix_geom_info (nside, stride, &w[0], &geom)
	return map_info_from_geom(geom)

def map_info_gauss_legendre(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Gauss-Legendre pixelization. Optimal
	weights are computed automatically. The pixels are laid out in nrings rings of
	constant colatitude, each with nphi pixels equally spaced in longitude, with the
	first pixel in each ring at longitude phi0."""
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_gauss_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom(geom)

def map_info_clenshaw_curtis(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Cylindrical Equi-rectangular pixelization
	with nrings iso-colatitude rings with nphi pixels each, such that the first and last
	rings have colatitude 0 and pi respectively. This corresponds to Clenshaw-Curtis
	quadrature."""
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_cc_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom(geom)

def map_info_fejer1(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Cylindrical Equi-rectangular pixelization
	with nrings iso-colatitude rings with nphi pixels each, such that the first and last
	rings have colatitude 0.5*pi/nrings and pi-0.5*pi/nrings respectively.
	This corresponds to Frejer's first quadrature."""
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_fejer1_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom(geom)

def map_info_fejer2(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Cylindrical Equi-rectangular pixelization
	with nrings iso-colatitude rings with nphi pixels each, such that the first and last
	rings have colatitude pi/(2*nrings-1) and pi respectively.
	This corresponds to Frejer's second quadrature.
	This function does *NOT* define any quadrature weights!"""
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_fejer2_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom(geom)

def map_info_mw(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_mw_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom(geom)

cdef map_info_from_geom(csharp.sharp_geom_info * geom):
	"""Constructs a map_info from a gemoetry pointer."""
	cdef int pair
	cdef int ring = 0
	cdef int npairs = geom.npairs
	cdef np.ndarray[np.float64_t,ndim=1] theta   = np.empty(2*npairs,dtype=np.float64)
	cdef np.ndarray[np.float64_t,ndim=1] phi0    = np.empty(2*npairs,dtype=np.float64)
	cdef np.ndarray[np.float64_t,ndim=1] weight  = np.empty(2*npairs,dtype=np.float64)
	cdef np.ndarray[np.int32_t,ndim=1]   nphi    = np.empty(2*npairs,dtype=np.int32)
	cdef np.ndarray[np.int32_t,ndim=1]   stride  = np.empty(2*npairs,dtype=np.int32)
	cdef np.ndarray[np.int64_t,ndim=1]   offsets = np.empty(2*npairs,dtype=np.int64)
	cdef csharp.sharp_ringinfo * info
	# This should have been as simple as for info in (geom.pair[pair].r1,geom.pair[pair].r2)
	cdef np.uintp_t infop
	cdef np.ndarray[np.uintp_t,ndim=1] tmp = np.empty(2,dtype=np.uintp)
	for pair in range(npairs):
		tmp[0] = <np.uintp_t>&geom.pair[pair].r1
		tmp[1] = <np.uintp_t>&geom.pair[pair].r2
		for infop in tmp:
			info = <csharp.sharp_ringinfo*> infop
			if info.nph >= 0:
				theta[ring]  = atan2(info.sth,info.cth)
				phi0[ring]   = info.phi0
				weight[ring] = info.weight
				nphi[ring]   = info.nph
				stride[ring] = info.stride
				offsets[ring]= info.ofs
				ring += 1
	cdef np.ndarray[np.int_t,ndim=1] order = np.argsort(offsets[:ring])
	return map_info(theta[order],nphi[order],phi0[order],offsets[order],stride[order],weight[order])

cdef class alm_info:
	cdef csharp.sharp_alm_info * info
	cdef readonly int lmax
	cdef readonly int mmax
	cdef readonly int stride
	cdef readonly int nelem
	cdef readonly np.ndarray mstart
	def __cinit__(self, lmax, mmax=None, stride=1, layout="triangular"):
		"""Constructs a new sharp spherical harmonic coefficient layout information
		for the given lmax and mmax. The layout defaults to triangular, but
		can be changed by explicitly specifying layout, either as a string
		naming layout (triangular or rectangular), or as an array containing the
		index of the first l for each m. Once constructed, an alm_info is immutable.
		The layouts are all m-major, with all the ls for each m consecutive."""
		if mmax is None: mmax = lmax
		if isinstance(layout,basestring):
			if layout == "triangular" or layout == "tri":
				m = np.arange(mmax+1)
				mstart = np.concatenate([[0],np.cumsum(lmax+1-np.arange(mmax))])*stride
			elif layout == "rectangular" or layout == "rect":
				mstart = np.arange(mmax+1)*(lmax+1)*stride
			else:
				raise ValueError("unkonwn layout: %s" % layout)
		else:
			mstart = layout
		self.info  = make_alm_helper(lmax,mmax,stride,mstart)
		self.lmax  = lmax
		self.mmax  = mmax
		self.stride= stride
		self.nelem = np.max(mstart + ((lmax+1)-np.arange(mmax+1))*stride)
		self.mstart= mstart
		self.mstart.flags.writeable = False
	def lm2ind(self, np.ndarray[int,ndim=1] l,np.ndarray[int,ndim=1] m):
		return self.mstart[m]+l*self.stride
	def transpose_alm(self, alm, out=None):
		"""In order to accomodate l-major ordering, which is not directoy
		supported by sharp, this function efficiently transposes Alm into
		Aml. If the out argument is specified, the transposed result will
		be written there. In order to perform an in-place transpose, call
		this function with the same array as "alm" and "out". If the out
		argument is not specified, then a new array will be constructed
		and returned."""
		if out is None: out = alm.copy()
		if out.dtype == np.complex128:
			self.transpose_alm_dp(out)
		else:
			self.transpose_alm_sp(out)
		return out
	cdef transpose_alm_dp(self,np.ndarray[np.complex128_t,ndim=1] alm):
		cdef int l=0
		cdef int m=0
		cdef int i
		cdef int j
		cdef np.complex128_t v
		cdef np.ndarray[int,ndim=1] mstart = self.mstart
		for i in range(alm.size):
			m += 1
			if m > self.mmax or m > l:
				l += 1
				m = 0
			j = mstart[m]+l*self.stride
			if j > i:
				v = alm[i]
				alm[i] = alm[j]
				alm[j] = v
	cdef transpose_alm_sp(self,np.ndarray[np.complex64_t,ndim=1] alm):
		cdef int l=0
		cdef int m=0
		cdef int i
		cdef int j
		cdef np.complex64_t v
		cdef np.ndarray[int,ndim=1] mstart = self.mstart
		for i in range(alm.size):
			m += 1
			if m > self.mmax or m > l:
				l += 1
				m = 0
			j = mstart[m]+l*self.stride
			if j > i:
				v = alm[i]
				alm[i] = alm[j]
				alm[j] = v
	def __dealloc__(self):
		csharp.sharp_destroy_alm_info(self.info)

cdef class sht:
	cdef public map_info minfo
	cdef public alm_info ainfo
	def __cinit__(self, minfo, ainfo):
		"""Construct a sharp Spherical Harmonics Transform (SHT) object, which
		transforms between maps with pixellication given by the map_info "minfo"
		and spherical harmonic coefficents given by alm_info "ainfo"."""
		self.minfo, self.ainfo = minfo, ainfo
	def alm2map(self, alm, map=None, spin=0):
		"""Transform the given spherical harmonic coefficients "alm" into
		a map space. If a map is specified as the "map" argument, output
		will be written there. Otherwise, a new map will be constructed
		and returned. "alm" has dimensions [ntrans,nspin,nalm], or
		[nspin,nalm] or [nalm] where ntrans is the number of independent
		transforms to perform in parallel, nspin is the number of spin
		components per transform (1 or 2), and nalm is the number of coefficients
		per alm."""
		alm = np.asarray(alm)
		ntrans, nspin = dim_helper(alm, "alm")
		# Create a compatible output map
		if map is None:
			map = np.empty([ntrans,nspin,self.minfo.npix],dtype=alm.real.dtype)
			map = map.reshape(alm.shape[:-1]+(map.shape[-1],))
		else:
			assert(alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree")
		execute(csharp.SHARP_ALM2MAP, self.ainfo, alm, self.minfo, map, spin=spin)
		return map
	def map2alm(self, map, alm=None, spin=0):
		"""Transform the given map "map" into a harmonic space. If "alm" is specified,
		output will be written there. Otherwise, a new alm will be constructed
		and returned. "map" has dimensions [ntrans,nspin,npix], or
		[nspin,npix] or [npix] where ntrans is the number of independent
		transforms to perform in parallel, nspin is the number of spin
		components per transform (1 or 2), and npix is the number of pixels per map."""
		map = np.asarray(map, dtype=np.float64)
		ntrans, nspin = dim_helper(map, "map")
		# Create a compatible output map
		if alm is None:
			alm = np.empty([ntrans,nspin,self.ainfo.nelem],dtype=np.result_type(map.dtype,0j))
			alm = alm.reshape(map.shape[:-1]+(alm.shape[-1],))
		else:
			assert(alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree")
		execute(csharp.SHARP_MAP2ALM, self.ainfo, alm, self.minfo, map, spin=spin)
		return alm

# alm and map have the formats:
#  [nlm]:           spin=0,    ntrans=1
#  [ns,nlm]:        spin=ns>1, ntrans=1
#  [ntrans,ns,nlm]: spin=ns>1, ntrans=ntrans
# So to do many spin 0 transforms in parallel, you would pass alm with
# the shape [:,1,:], which can be created from a 2d alm by alm[:,None]
def dim_helper(a, name):
	assert(a.ndim > 3 and a.ndim <= 3, name + " must be [nlm], [ntrf*ncomp,nlm] or [ntrf,ncomp,nlm]")
	if a.ndim == 1:
		ntrans, nspin = 1, 1
	elif a.ndim == 2:
		ntrans, nspin = 1, a.shape[0]
	elif a.ndim == 3:
		ntrans, nspin = a.shape[:2]
	assert(nspin < 3, name + " spin axis must have length 1 or 2 (T and P must be done separately)")
	return ntrans, nspin

def execute(type, alm_info ainfo, alm, map_info minfo, map, spin):
	assert(isinstance(alm, np.ndarray), "alm must be a numpy array")
	assert(isinstance(map, np.ndarray), "map must be a numpy array")
	cdef int i
	ntrans, nspin = dim_helper(alm, "alm")
	assert(spin == 0 and nspin != 1 or spin > 0 and nspin != 2,
		"Dimension -2 of maps and alms must be 2 for spin transforms and 1 for scalar transforms.")
	try:
		type = typemap[type]
	except KeyError:
		pass
	alm3 = alm.reshape(ntrans,nspin,-1)
	map3 = map.reshape(ntrans,nspin,-1)
	if map.dtype == np.float64:
		execute_dp(type, spin, ainfo, alm3, minfo, map3)
	else:
		execute_sp(type, spin, ainfo, alm3, minfo, map3)

cpdef execute_dp(int type, int spin, alm_info ainfo, np.ndarray[np.complex128_t,ndim=3,mode="c"] alm, map_info minfo, np.ndarray[np.float64_t,ndim=3,mode="c"] map):
	cdef int ntrans = map.shape[0]
	cdef int nspin  = map.shape[1]
	cdef n = ntrans*nspin
	cdef np.ndarray[np.complex128_t,ndim=2,mode="c"] aflat = alm.reshape(n,alm.shape[2])
	cdef np.ndarray[np.float64_t,ndim=2,mode="c"]    mflat = map.reshape(n,map.shape[2])
	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]      aptrs = np.empty(n,dtype=np.uintp)
	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]      mptrs = np.empty(n,dtype=np.uintp)
	cdef int i
	for i in range(n):
		aptrs[i] = <np.uintp_t>&aflat[i,0]
		mptrs[i] = <np.uintp_t>&mflat[i,0]
	execute_helper(type, ainfo, aptrs, minfo, mptrs, spin, ntrans, csharp.SHARP_DP)

cpdef execute_sp(int type, int spin, alm_info ainfo, np.ndarray[np.complex64_t,ndim=3,mode="c"] alm, map_info minfo, np.ndarray[np.float32_t,ndim=3,mode="c"] map):
	cdef int ntrans = map.shape[0]
	cdef int nspin  = map.shape[1]
	cdef n = ntrans*nspin
	cdef np.ndarray[np.complex64_t,ndim=2,mode="c"]  aflat = alm.reshape(n,alm.shape[2])
	cdef np.ndarray[np.float32_t,ndim=2,mode="c"]    mflat = map.reshape(n,map.shape[2])
	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]      aptrs = np.empty(n,dtype=np.uintp)
	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]      mptrs = np.empty(n,dtype=np.uintp)
	cdef int i
	for i in range(n):
		aptrs[i] = <np.uintp_t>&aflat[i,0]
		mptrs[i] = <np.uintp_t>&mflat[i,0]
	execute_helper(type, ainfo, aptrs, minfo, mptrs, spin, ntrans, 0)

typemap = { "map2alm": csharp.SHARP_MAP2ALM, "alm2map": csharp.SHARP_ALM2MAP, "alm2map_deriv1": csharp.SHARP_ALM2MAP_DERIV1 }

cdef execute_helper(int type,
		alm_info ainfo, np.ndarray[np.uintp_t,ndim=1] alm,
		map_info minfo, np.ndarray[np.uintp_t,ndim=1] map,
		int spin=0, int ntrans=1, int flags=0):
	csharp.sharp_execute(type, spin, <void*>&alm[0], <void*>&map[0],
			minfo.geom, ainfo.info, ntrans, flags, NULL, NULL)

cdef csharp.sharp_geom_info * make_geometry_helper(
		int ntheta,
		np.ndarray[int,ndim=1] nphi,
		np.ndarray[csharp.ptrdiff_t,ndim=1] offsets,
		np.ndarray[int,ndim=1] stride,
		np.ndarray[double,ndim=1] phi0,
		np.ndarray[double,ndim=1] theta,
		np.ndarray[double,ndim=1] weight):
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_geom_info(ntheta, &nphi[0], &offsets[0], &stride[0], &phi0[0], &theta[0], &weight[0], &geom)
	return geom

cdef csharp.sharp_alm_info * make_alm_helper(int lmax, int mmax, int stride, np.ndarray[csharp.ptrdiff_t,ndim=1] mstart):
	cdef csharp.sharp_alm_info * info
	csharp.sharp_make_alm_info(lmax, mmax, stride, &mstart[0], &info)
	return info

cdef csharp.sharp_alm_info * make_triangular_alm_helper(int lmax, int mmax, int stride):
	cdef csharp.sharp_alm_info * info
	csharp.sharp_make_triangular_alm_info(lmax, mmax, stride, &info)
	return info
