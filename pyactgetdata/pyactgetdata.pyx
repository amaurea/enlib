import cython
import numpy as np
cimport numpy as np
cimport cactgetdata
from libc.stdint cimport uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.stdio  cimport printf

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
	cdef str _fname
	def __cinit__(self, fname=None):
		"""Construdt a dirfile object by opening the specified file fname"""
		self.dfile = NULL
		self.fieldinfo = None
		self._fname = None
		if fname is not None: self.open(fname)
	def open(self, fname):
		if self.is_open(): self.close()
		self.dfile = cactgetdata.ACTpolDirfile_open(fname)
		self._fname = fname
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
		return sorted(self.fieldinfo.keys())
	@property
	def nfield(self):
		return len(self.fieldinfo)
	@property
	def fname(self):
		return self._fname
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
	@cython.boundscheck(False)
	def getdata_multi(self, field_list, field_type=None, nthread=0):
		"""Parallel version of getdata. field_list is a list of field names, all of
		which must have the same length and type as the output_array's last dimension.
		output_array is a 2d array [len(field_list),nsamp] that the data read will
		be written to. nthread controls the number of omp threads.
		The default, 0, lets OMP decide."""
		if not self.is_open(): raise IOError("Dfile is not open")
		if len(field_list) == 0: return None
		for field in field_list:
			if not field in self.fieldinfo:
				raise IOError("Field %s does not exist" % field)

		# Collect information needed to build the output array. The type is simple
		if field_type is None: field_type = self.native_type(field)
		dtype = typemap[field_type]()
		cdef char * tmp = field_type
		cdef char ctype = tmp[0]
		# Read the first field to get the length
		first_row = self.getdata(field_list[0], field_type)
		cdef int nsamp = first_row.size
		cdef int nbyte = nsamp*dtype.nbytes
		cdef int nfield= len(field_list)
		# We can now construct the output array
		cdef np.ndarray[np.uint8_t,ndim=2,mode="c"] arr  = np.empty((nfield,nbyte),dtype=np.uint8)
		# Construct pointers to each row
		cdef void ** rows = <void**> malloc(nfield*sizeof(void*));
		for i in range(nfield): rows[i] = <void*>&arr[i,0]
		# And a C list of field names
		cdef char ** cnames = <char**> malloc(nfield*sizeof(char*))
		for i in range(nfield): cnames[i] = field_list[i]
		# We are now ready to read the data
		cactgetdata.read_channels_into_omp(nfield, nthread, ctype, nbyte, self.dfile, cnames, rows)
		# And return
		free(rows)
		free(cnames)
		return arr.view(dtype)
	def __enter__(self):
		return self
	def __exit__(self, type, value, traceback):
		self.close()
