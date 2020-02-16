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

def encode(s):
	if isinstance(s, unicode):
		return s.encode("utf8")
	else:
		return s

def decode(s):
	if str is unicode:
		try:
			return (<bytes>s).decode("utf8")
		except UnicodeDecodeError as e:
			return str(s)
	else:
		return s

def tochar(s):
	try: return ord(s)
	except TypeError: return s

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
		if fname is not None: self.open(fname)
	def open(self, fname):
		if self.is_open(): self.close()
		tmp = encode(fname)
		cdef char * cfname = tmp
		self.dfile = cactgetdata.ACTpolDirfile_open(cfname)
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
	def native_type(self, field): return self.fieldinfo[field][1]
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
			res[decode(field)] = (decode(category), chr(type))
		return res
	@cython.boundscheck(False)
	def getdata(self, field, type=None):
		if not self.is_open(): raise IOError("Dfile is not open")
		tmp = encode(field)
		cdef char * cfield = tmp
		exists = cactgetdata.ACTpolDirfile_has_channel(self.dfile, cfield)
		if not exists: raise IOError("Field %s does not exist" % field)
		if type is None: type = self.native_type(field)
		dtype = typemap[type]()
		cdef int nsamp
		cdef np.npy_intp size, i
		tmp2 = encode(type)
		cdef char ctype = tochar(tmp2[0])
		# This sadly involves a copy, but avoiding that did not work
		cdef uint8_t * data = <uint8_t*>cactgetdata.ACTpolDirfile_read_channel(ctype, self.dfile, cfield, &nsamp)
		size = nsamp*dtype.nbytes
		cdef np.ndarray[np.uint8_t,ndim=1] res = np.empty([size],dtype=np.uint8)
		for i in range(size): res[i] = data[i]
		free(data)
		return res.view(typemap[type])
	@cython.boundscheck(False)
	def getdata_multi(self, field_list, field_type=None, int nthread=0):
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
		tmps = [encode(entry) for entry in field_list]
		for i in range(nfield): cnames[i] = tmps[i]
		tmp = encode(field_type)
		cdef char ctype = tochar(tmp[0])
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

