import cython
import numpy as np
cimport numpy as np
cimport cactgetdata
from libc.stdint cimport int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
from libc.stdlib cimport malloc, free
cdef extern from "numpy/arrayobject.h":
	void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

typemap = {
	"s":np.int16,
	"u":np.uint16,
	"S":np.int32,
	"U":np.uint32,
	"f":np.float32,
	"d":np.float64
}
typesize = {
	"s":2,
	"u":2,
	"S":4,
	"U":4,
	"f":4,
	"d":8,
}

# Basic passthrough first
cdef class dirfile:
	"""This class is a wrapper for the ACTpolDirfile class."""
	cdef cactgetdata.ACTpolDirfile * dfile
	def __cinit__(self, fname=None):
		"""Construdt a dirfile object by opening the specified file fname"""
		self.dfile = NULL
		if fname is not None: self.open(fname)
	def open(self, fname):
		if self.is_open(): self.close()
		self.dfile = cactgetdata.ACTpolDirfile_open(fname)
		if self.dfile is NULL:
			raise IOError("Error opening dirfile '%s'" % fname)
	def is_open(self): return self.dfile is not NULL
	def close(self):
		if self.is_open():
			cactgetdata.ACTpolDirfile_close(self.dfile)
			self.dfile = NULL
	def __dealloc__(self):
		self.close()
	def getdata(self, char * field, char * type):
		if not self.is_open(): raise IOError("Dfile is not open")
		exists = cactgetdata.ACTpolDirfile_has_channel(self.dfile, field)
		if not exists: raise IOError("Field %s does not exist" % field)
		cdef int nsamp
		cdef np.npy_intp size
		cdef void * data
		#cdef np.ndarray[np.uint8_t,ndim=1] res
		data = cactgetdata.ACTpolDirfile_read_channel(type[0], self.dfile, field, &nsamp)
		size = nsamp*typesize[type]
		return nsamp, size
		#res  = np.PyArray_SimpleNewFromData(1, &size, np.NPY_INT8, data)
		## It's supposed to be possible to use PyArray_ENABLEFLAGS to
		## make numpy claim ownership of the data. But that gives me linking
		## problems.
		##PyArray_ENABLEFLAGS(res, np.NPY_OWNDATA)
		#res  = np.array(res)
		#free(data)
		#return res.view(typemap[type])
