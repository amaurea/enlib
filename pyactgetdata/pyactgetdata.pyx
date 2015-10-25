import cython
import numpy as np
cimport numpy as np
cimport cactgetdata
from libc.stdint cimport uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
from libc.stdlib cimport malloc, free

typemap = {
	"s":np.int16,
	"u":np.uint16,
	"S":np.int32,
	"U":np.uint32,
	"f":np.float32,
	"d":np.float64
}

# Basic passthrough first
cdef class dirfile:
	"""This class is a wrapper for the ACTpolDirfile class."""
	cdef cactgetdata.ACTpolDirfile * dfile
	cdef dict fieldinfo
	def __cinit__(self, fname=None):
		"""Construdt a dirfile object by opening the specified file fname"""
		self.dfile = NULL
		self.fieldinfo = None
		if fname is not None: self.open(fname)
	def open(self, fname):
		if self.is_open(): self.close()
		self.dfile = cactgetdata.ACTpolDirfile_open(fname)
		if self.dfile is NULL:
			raise IOError("Error opening dirfile '%s'" % fname)
		self.fieldinfo = self._list_()
	def is_open(self): return self.dfile is not NULL
	def close(self):
		if self.is_open():
			cactgetdata.ACTpolDirfile_close(self.dfile)
			self.dfile = NULL
			self.fieldinfo = None
	def __dealloc__(self):
		self.close()
	@property
	def fields(self):
		return self.fieldinfo.keys()
	@property
	def nfield(self):
		return len(self.fieldinfo)
	def category(self, field): return self.fieldinfo[field][0]
	def native_type(self, field): return chr(self.fieldinfo[field][1])
	def _list_(self):
		cdef int i, n, status
		cdef char * category
		cdef char * field
		cdef char type
		n = cactgetdata.GetNEntry(self.dfile.format)
		if n < 0: raise IOError("Invalid format field in dirfile")
		res = {}
		for i in range(n):
			status = cactgetdata.GetEntryInfo(self.dfile.format, i, &category, &field, &type)
			if not status: raise IOError("Error accessing field %d/%d" % (i,n))
			res[field] = (category, type)
		return res
	@cython.boundscheck(False)
	def getdata(self, char * field, type=None):
		if not self.is_open(): raise IOError("Dfile is not open")
		exists = cactgetdata.ACTpolDirfile_has_channel(self.dfile, field)
		if not exists: raise IOError("Field %s does not exist" % field)
		if type is None: type = self.native_type(field)
		dtype = typemap[type]()
		cdef char * ctype = type
		cdef int nsamp
		cdef np.npy_intp size, i
		# This sadly involves a copy, but avoiding that did not work
		cdef uint8_t * data = <uint8_t*>cactgetdata.ACTpolDirfile_read_channel(ctype[0], self.dfile, field, &nsamp)
		size = nsamp*dtype.nbytes
		cdef np.ndarray[np.uint8_t,ndim=1] res = np.empty([size],dtype=np.uint8)
		for i in range(size): res[i] = data[i]
		free(data)
		return res.view(typemap[ctype])
