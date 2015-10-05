import numpy as np, time
from enlib import config, fft, utils

config.default("gfilter_jon_naz", 8, "The number of azimuth modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_nt",  8, "The number of time modes to fit/subtract in Jon's polynomial ground filter.")
def filter_poly_jon(tod, az, naz=None, nt=None, deslope=False):
	"""Apply Jon's polynomial azimuth filter to tod, with the azimuth
	of each sample given by az. naz azimuth modes will be fit and subtracted.
	At the same time, nt polynomials in time will be fit but not subtracted."""
	naz = config.get("gfilter_jon_naz", naz)
	nt  = config.get("gfilter_jon_nt", nt)
	t1 = time.time()
	if naz <= 0: return tod # Don't do anything if we're not subtracting anything
	# d is a 2d view of tod, and should not involve any copies
	d = tod.reshape(-1,tod.shape[-1])
	# Construct basis
	ndet,nsamp = d.shape
	nmax       = max(naz,nt)
	params     = np.zeros([nmax,nsamp])
	for i in range(nmax): params[i,i] = 1
	polynomials = fft.ichebt(params)
	tbasis  = polynomials[:nt]
	# The azimuth basis is built by interpreting the [0:nsamp] range as
	# azimuth, and interpolating it into time domain. I do this because the
	# cheb transform needs evenly spaced points. It might be simpler to just
	# construct the basis directly via a recurrence relation.
	az_basis_map = (az-np.min(az))*nsamp/(np.max(az)-np.min(az))
	t2 = time.time()
	azbasis = utils.interpol(polynomials[1:naz], az_basis_map[None], order=1, mask_nan=False)
	basis = np.concatenate([azbasis,tbasis])
	t3 = time.time()
	# We will now solve jointly for coefficients for each of these. We treat each
	# detector as independent, and the noise as being white. The atmosphere is
	# instead handled via the t basis.
	b = basis.dot(d.T)
	A = basis.dot(basis.T)
	t4 = time.time()
	amps = np.linalg.solve(A,b)
	t5 = time.time()
	# Subtract az modes, but not t modes, which were just included in the fit to
	# avoid them leaking into az
	d -= amps[:naz].T.dot(basis[:naz])
	t6 = time.time()
	if deslope:
		# Since az there isn't necessarily a whole period of az swipes in a TOD, subtracting
		# az polynomials may induce non-periodicity. So deslope if we are going to do more stuff
		# in fourier space.
		utils.deslope(d, w=8, inplace=True)
	# We could just return tod here, but in case any copies were inadvertantly made,
	# we use d instead.
	return d.reshape(tod.shape)

def filter_poly_jon2(tod, az, naz=None, nt=None, deslope=True):
	"""Fix naz Legendre polynomials in az and nt other polynomials
	in t jointly. Then subtract the best fit from the data.
	The subtraction is inplace, so tod is modified. If naz or nt are
	negative, they are fit for, but not subtracted.
	NOTE: This function may leave tod nonperiodic.
	"""
	naz = config.get("gfilter_jon_naz", naz)
	nt  = config.get("gfilter_jon_nt", nt)
	naz, asign = np.abs(naz), np.sign(naz)
	nt,  tsign = np.abs(nt),  np.sign(nt)
	d   = tod.reshape(-1,tod.shape[-1])
	if naz == 0 and nt == 0: return tod
	B = np.zeros([naz+nt,d.shape[-1]],dtype=tod.dtype)
	if naz > 0:
		# Build azimuth basis as Legendre polynomials
		x = utils.rescale(az,[-1,1])
		B[0] = x
		if naz > 1: B[1] = 1.5*x**2-0.5
		for i in range(2, naz):
			B[i] = ((2*i+1)*x*B[i-1] - i*B[i-2])/(i+1)
	if nt > 0:
		x = np.linspace(-1,1,d.shape[-1],endpoint=False)
		for i in range(nt): B[naz+i] = x**i
	# Solve for the best fit for each detector, [nbasis,ndet]
	amps = np.linalg.solve(B.dot(B.T),B.dot(d.T))
	# Subtract the best fit
	if asign > 0: d -= amps[:naz].T.dot(B[:naz])
	if tsign > 0: d -= amps[naz:naz+nt].T.dot(B[naz:naz+nt])
	if deslope: utils.deslope(d, w=8, inplace=True)
	return d.reshape(tod.shape)

def filter_poly_jon_weighted(tod, az, weights=None, naz=None, nt=None, deslope=True):
	"""Fix naz Legendre polynomials in az and nt other polynomials
	in t jointly. Then subtract the best fit from the data.
	The subtraction is inplace, so tod is modified. If naz or nt are
	negative, they are fit for, but not subtracted.
	NOTE: This function may leave tod nonperiodic.
	"""
	naz = config.get("gfilter_jon_naz", naz)
	nt  = config.get("gfilter_jon_nt", nt)
	naz, asign = np.abs(naz), np.sign(naz)
	nt,  tsign = np.abs(nt),  np.sign(nt)
	d  = tod.reshape(-1,tod.shape[-1])
	if naz == 0 and nt == 0: return tod
	B = np.zeros([naz+nt,d.shape[-1]],dtype=tod.dtype)
	if naz > 0:
		# Build azimuth basis as Legendre polynomials
		x = utils.rescale(az,[-1,1])
		B[0] = x
		if naz > 1: B[1] = 1.5*x**2-0.5
		for i in range(2, naz):
			B[i] = ((2*i+1)*x*B[i-1] - i*B[i-2])/(i+1)
	if nt > 0:
		x = np.linspace(-1,1,d.shape[-1],endpoint=False)
		for i in range(nt): B[naz+i] = x**i
	# Solve for the best fit for each detector, [nbasis,ndet]
	if weights is None:
		amps = np.linalg.solve(B.dot(B.T),B.dot(d.T))
	else:
		w = weights.reshape(-1,weights.shape[-1])
		amps = np.zeros([naz+nt,d.shape[0]],dtype=tod.dtype)
		for di in range(len(tod)):
			amps[:,di] = np.linalg.solve(B.dot(w[di,:,None]*B.T),B.dot(w[di]*d[di]))
	if asign > 0: d -= amps[:naz].T.dot(B[:naz])
	if tsign > 0: d -= amps[naz:naz+nt].T.dot(B[naz:naz+nt])
	if deslope: utils.deslope(d, w=8, inplace=True)
	return d.reshape(tod.shape)
