import numpy as np
cimport numpy as np
from libc.stdlib cimport free
cdef extern from "read_table.h":
	int read_table(char * filename, char * delim, char * comments, double ** arr, int * dims)

def loadtxt(fname, dtype=np.float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0):
	if usecols != None or skiprows != 0 or converters != None:
		# These options aren't implemented yet, so fall back on standard
		# loadtxt if they are specified.
		return np.loadtxt(fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin)
	if delimiter == None: delimiter = " \t"
	# Read the table and convert it into a numpy array
	cdef np.ndarray[int] dims = np.zeros([2],dtype=np.int32)
	cdef double * arr = NULL
	cdef int stat
	cdef int r
	cdef int c
	cdef int i
	cdef int * foo = NULL
	stat = read_table(fname, delimiter, comments, &arr, &dims[0])
	if not stat: raise ValueError("Invalid element in ascii table")
	cdef np.ndarray[double,ndim=2] res = np.empty(dims,dtype=np.float)
	i = 0
	for r in range(dims[0]):
		for c in range(dims[1]):
			res[r,c] = arr[i]
			i += 1
	free(arr)
	res = np.squeeze(res)
	if unpack:
		res = res.T
	return res
