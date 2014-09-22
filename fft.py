"""This is a convenience wrapper of pyfftw."""
import numpy as np, pyfftw, multiprocessing, os, enlib.utils
from enlib import log

try:
	nthread_fft = int(os.environ['OMP_NUM_THREADS'])
except KeyError, ValueError:
	nthread_fft=multiprocessing.cpu_count()
nthread_ifft=nthread_fft

def fft(tod, ft=None, nthread=0, axes=[-1]):
	"""Compute discrete fourier transform of tod, and store it in ft. What
	transform to do (real or complex, number of dimension etc.) is determined
	from the size and type of tod and ft. The optional nthread argument specifies
	the number of theads to use in the fft. The default (0) uses the value specified
	by the OMP_NUM_THREAD environment varible if that is specified, or the total
	number of cores on the computer otherwise. If ft is left out, a complex
	transform is assumed."""
	tod = asfcarray(tod)
	if tod.size == 0: return
	nt = nthread or nthread_fft
	if ft is None:
		otype = np.result_type(tod.dtype,0j)
		ft  = np.empty(tod.shape, otype)
		tod = tod.astype(otype, copy=False)
	plan = pyfftw.FFTW(tod, ft, flags=['FFTW_ESTIMATE'], threads=nt, axes=axes)
	plan()
	return ft

def ifft(ft, tod=None, nthread=0, normalize=False, axes=[-1]):
	"""Compute inverse discrete fourier transform of ft, and store it in tod. What
	transform to do (real or complex, number of dimension etc.) is determined
	from the size and type of tod and ft. The optional nthread argument specifies
	the number of theads to use in the fft. The default (0) uses the value specified
	by the OMP_NUM_THREAD environment varible if that is specified, or the total
	number of cores on the computer otherwise. By default this is not nrmalized,
	meaning that fft followed by ifft will multiply the data by the length of the
	transform. By specifying the normalize argument, you can turn normalization
	on, though the normalization step will not use paralellization."""
	ft = asfcarray(ft)
	if ft.size == 0: return
	nt = nthread or nthread_ifft
	if tod is None: tod = np.empty(ft.shape, ft.dtype)
	plan = pyfftw.FFTW(ft, tod, flags=['FFTW_ESTIMATE'], direction='FFTW_BACKWARD', threads=nt, axes=axes)
	plan(normalise_idft=normalize)
	return tod

def rfft(tod, ft=None, nthread=0, axes=[-1]):
	"""Equivalent to fft, except that if ft is not passed, it is allocated with
	appropriate shape and data type for a real-to-complex transform."""
	tod = asfcarray(tod)
	if ft is None:
		oshape = list(tod.shape)
		oshape[axes[-1]] = oshape[axes[-1]]/2+1
		dtype = np.result_type(tod.dtype,0j)
		ft = np.empty(oshape, dtype)
	return fft(tod, ft, nthread, axes)

def irfft(ft, tod=None, n=None, nthread=0, normalize=False, axes=[-1]):
	"""Equivalent to ifft, except that if tod is not passed, it is allocated with
	appropriate shape and data type for a complex-to-real transform. If n
	is specified, that is used as the length of the last transform axis
	of the output array. Otherwise, the length of this axis is computed
	assuming an even original array."""
	ft = asfcarray(ft)
	if tod is None:
		oshape = list(ft.shape)
		oshape[axes[-1]] = n or (oshape[axes[-1]]-1)*2
		dtype = np.zeros([],ft.dtype).real.dtype
		tod = np.empty(oshape, dtype)
	return ifft(ft, tod, nthread, normalize, axes)

def redft00(a, b=None, nthread=0, normalize=False):
	"""pyFFTW does not support the DCT yet, so this is a work-around.
	It's not very fast, sadly - about 5 times slower than an rfft.
	Transforms along the last axis."""
	a = asfcarray(a)
	if b is None: b = np.empty(a.shape, a.dtype)
	n = a.shape[-1]
	tshape = a.shape[:-1] + (2*(n-1),)
	itmp = np.empty(tshape, a.dtype)
	itmp[...,:n] = a[...,:n]
	itmp[...,n:] = a[...,-2:0:-1]
	otmp = rfft(itmp, axes=[-1], nthread=nthread)
	del itmp
	b[...] = otmp[...,:n].real
	if normalize: b /= 2*(n-1)
	return b

def chebt(a, b=None, nthread=0):
	"""The chebyshev transform of a, along its last dimension."""
	b = redft00(a, b, nthread, normalize=True)
	b[1:-1] *= 2
	return b

def ichebt(a, b=None, nthread=0):
	"""The inverse chebyshev transform of a, along its last dimension."""
	a = asfcarray(a).copy()
	a[1:-1] *= 0.5
	return redft00(a, b, nthread)

def fft_len(n, factors=[2,3,5,7,11,13]):
	return enlib.utils.nearest_product(n, factors)

def asfcarray(a):
	a = np.asarray(a)
	return np.asarray(a, np.result_type(a,0.0))
