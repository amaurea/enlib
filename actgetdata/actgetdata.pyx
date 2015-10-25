import cython
import numpy as np
cimport numpy as np
cimport cactgetdata

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
		self.fname = fname
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
	def list(self):
		"""List contained fields."""
		if not self.is_open():
			raise IOError("dirfile is not open")
		cdef cactgetdata.FormatType * format = self.dfile.format
		fields = {}
		for i in range(format.n_raw):
			fields[format.rawEntries[i].field] = ("raw",format.rawEntries[i].type)
		for i in range(format.n_lincom):
			fields[format.lincomEntries[i].field] = ("lincom", "d")
		for i in range(format.n_linterp):
			fields[format.linterpEntries[i].field] = ("derived", "d")
		for i in range(format.n_multiply):
			fields[format.multiplyEntries[i].field] = ("derived", "d")
		for i in range(format.n_mplex):
			fields[format.mplexEntries[i].field] = ("derived", "d")
		for i in range(format.n_bit):
			fields[format.bitEntries[i].field] = ("derived", "u")
		return fields
