import numpy as np
cimport numpy as np
cimport csharp

cdef class map_info:
	cdef csharp.sharp_geom_info * geom
	cdef public int nrow
	cdef public int npix
	def __cinit__(self, theta, nphi=0, phi0=0, offsets=None, stride=None, weight=None):
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
		self.nrow = ntheta
		self.npix = np.sum(nphi)
	def __dealloc__(self):
		csharp.sharp_destroy_geom_info(self.geom)

cdef class alm_info:
	cdef csharp.sharp_alm_info * info
	cdef public int lmax
	cdef public int mmax
	cdef public int nelem
	def __cinit__(self, lmax, mmax=None, stride=1, mstart=None):
		if mmax is None: mmax = lmax
		if mstart is None:
			self.info = make_triangular_alm_helper(lmax, mmax, stride)
			self.nelem = (lmax+1)*(lmax+2)/2 - (lmax-mmax)*(lmax-mmax+1)/2
		else:
			self.info  = make_alm_helper(lmax,mmax,stride, mstart)
			self.nelem = np.max(mstart)+lmax+1
		self.lmax  = lmax
		self.mmax  = mmax
	def __dealloc__(self):
		csharp.sharp_destroy_alm_info(self.info)

cdef class sht:
	cdef public map_info minfo
	cdef public alm_info ainfo
	def __cinit__(self, minfo, ainfo):
		self.minfo, self.ainfo = minfo, ainfo
	cpdef alm2map(self, alm, map=None):
		alm = np.asarray(alm)
		ntrans, nspin = dim_helper(alm, "alm")
		# Create a compatible output map
		if map is None:
			map = np.empty([ntrans,nspin,self.minfo.npix],dtype=alm.real.dtype)
			map = map.reshape(alm.shape[:-1]+(map.shape[-1],))
		else:
			assert(alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree")
		execute(csharp.SHARP_ALM2MAP, self.ainfo, alm, self.minfo, map)
		return map
	cpdef map2alm(self, map, alm=None):
		map = np.asarray(map, dtype=np.float64)
		ntrans, nspin = dim_helper(map, "map")
		# Create a compatible output map
		if alm is None:
			alm = np.empty([ntrans,nspin,self.ainfo.nelem],dtype=np.result_type(map.dtype,0j))
			alm = alm.reshape(map.shape[:-1]+(alm.shape[-1],))
		else:
			assert(alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree")
		execute(csharp.SHARP_MAP2ALM, self.ainfo, alm, self.minfo, map)
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

def execute(type, alm_info ainfo, alm, map_info minfo, map):
	assert(isinstance(alm, np.ndarray), "alm must be a numpy array")
	assert(isinstance(map, np.ndarray), "map must be a numpy array")
	cdef int i
	ntrans, nspin = dim_helper(alm, "alm")
	try:
		type = typemap[type]
	except KeyError:
		pass
	alm3 = alm.reshape(ntrans,nspin,-1)
	map3 = map.reshape(ntrans,nspin,-1)
	if map.dtype == np.float64:
		execute_dp(type, ainfo, alm3, minfo, map3)
	else:
		execute_sp(type, ainfo, alm3, minfo, map3)

cpdef execute_dp(int type, alm_info ainfo, np.ndarray[np.complex128_t,ndim=3,mode="c"] alm, map_info minfo, np.ndarray[np.float64_t,ndim=3,mode="c"] map):
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
	execute_helper(type, ainfo, aptrs, minfo, mptrs, nspin>1, ntrans, csharp.SHARP_DP)

cpdef execute_sp(int type, alm_info ainfo, np.ndarray[np.complex64_t,ndim=3,mode="c"] alm, map_info minfo, np.ndarray[np.float32_t,ndim=3,mode="c"] map):
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
	execute_helper(type, ainfo, aptrs, minfo, mptrs, nspin>1, ntrans, 0)

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
