import numpy as np
from enlib import utils, array_ops

# This sort of filtering isn't all that effective. Signal that
# obviously isn't scan-synchronous still gets picked up. Need
# to solve jointly or something.

def azfilter_basis(tod, weight, az, basis, nbin=100):
	# tod:    [ndet,nsamp]
	# weight: [ndet,nsamp]
	# basis:  [nbasis,ndet]
	# az:     [nsamp]
	az, inds, pix = make_bins(az, nbin)
	# Solve for az-binned signal for each basis function
	tod_rhs = basis.dot(tod*weight)
	tod_div = np.einsum("ad,bd,di->abi",basis,basis,weight)
	bin_rhs = bin_by_pix(tod_rhs, pix, nbin)
	bin_div = bin_by_pix(tod_div, pix, nbin)
	del tod_rhs, tod_div
	bin_sig = array_ops.solve_multi(bin_div, bin_rhs, axes=[0,1])
	# Interpolate model back to full resolution
	model = utils.interpol(bin_sig, inds)
	return tod - basis.T.dot(model)

def azfilter_individual(tod, az, nbin=5):
	# tod: [ndet,nsamp], az: [nsamp]
	tflat = tod.reshape(-1, tod.shape[-1])
	az, inds, pix = make_bins(az, nbin)
	bin_rhs = bin_by_pix(tflat, pix, nbin)
	bin_div = bin_by_pix(1, pix, nbin)
	bin_sig = bin_rhs/bin_div
	model   = utils.interpol(bin_sig, inds, order=3)
	return (tod - model).reshape(tod.shape), bin_sig

def make_bins(az, nbin):
	az   = utils.unwind(az)
	abox = np.array([np.min(az),np.max(az)])
	inds = (az-abox[0])/(abox[1]-abox[0])*nbin
	pix  = np.minimum(np.floor(inds).astype(np.int32),nbin-1)
	return az, inds[None], pix

def bin_by_pix(a, pix, nbin):
	a = np.asarray(a)
	if a.ndim == 0: a = np.full(len(pix), a)
	fa = a.reshape(-1,a.shape[-1])
	fo = np.zeros([fa.shape[0],nbin])
	for i in range(len(fa)):
		fo[i] = np.bincount(pix, fa[i], minlength=nbin)
	return fo.reshape(a.shape[:-1]+(nbin,))

