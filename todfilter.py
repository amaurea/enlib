import numpy as np, time, h5py
from enlib import config, fft, utils, gapfill

config.default("gfilter_jon_naz", 8, "The number of azimuth modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_nt",  10, "The number of time modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_nhwp", 0, "The number of hwp modes to fit/subtract in Jon's polynomial ground filter.")
config.default("gfilter_jon_niter", 3, "The number of time modes to fit/subtract in Jon's polynomial ground filter.")

config.default("gapfill", "linear", "TOD gapfill method. Can be 'copy' or 'linear'")
config.default("gapfill_context", 10, "Samples of context to use for matching up edges of cuts.")
def filter_poly_jon(tod, az, weights=None, naz=None, nt=None, niter=None, cuts=None, hwp=None, nhwp=None, deslope=True):
	"""Fix naz Legendre polynomials in az and nt other polynomials
	in t jointly. Then subtract the best fit from the data.
	The subtraction is inplace, so tod is modified. If naz or nt are
	negative, they are fit for, but not subtracted.
	NOTE: This function may leave tod nonperiodic.
	"""
	#moomoo = tod[:8].copy()
	naz = config.get("gfilter_jon_naz", naz)
	nt  = config.get("gfilter_jon_nt", nt)
	nhwp= config.get("gfilter_jon_nhwp", nhwp)
	niter = config.get("gfilter_jon_niter", niter)
	gapfiller = {"copy":gapfill.gapfill_copy, "linear":gapfill.gapfill_linear, "cubic":gapfill.gapfill_cubic}[config.get("gapfill")]
	context = config.get("gapfill_context")
	do_gapfill = cuts is not None
	#print "Mos", naz, nt, nhwp
	#print hwp
	# No point in iterating if we aren't gapfilling
	if not do_gapfill: niter = 1
	if hwp is None or np.all(hwp==0): nhwp = 0
	naz, asign = np.abs(naz), np.sign(naz)
	nt,  tsign = np.abs(nt),  np.sign(nt)
	nhwp,hsign = np.abs(nhwp),np.sign(nhwp)
	d   = tod.reshape(-1,tod.shape[-1])
	if naz == 0 and nt == 0 and nhwp == 0: return tod
	# Build our set of basis functions. These are shared
	# across iterations.
	B = np.zeros([naz+nt+nhwp,d.shape[-1]],dtype=tod.dtype)
	if naz > 0:
		# Build azimuth basis as polynomials
		x = utils.rescale(az,[-1,1])
		for i in range(naz): B[i] = x**(i+1)
	if nt > 0:
		x = np.linspace(-1,1,d.shape[-1],endpoint=False)
		for i in range(nt): B[naz+i] = x**i
	if nhwp > 0:
		# Use sin and cos to avoid discontinuities
		c = np.cos(hwp)
		s = np.sin(hwp)
		for i in range(nhwp):
			j = i/2+1
			x = np.cos(j*hwp) if i%2 == 0 else np.sin(j*hwp)
			B[naz+nt+i] = x
	for it in range(niter):
		if do_gapfill: gapfiller(d, cuts, inplace=True, overlap=context)
		# Solve for the best fit for each detector, [nbasis,ndet]
		# B[b,n], d[d,n], amps[b,d]
		if weights is None:
			amps = np.linalg.solve(B.dot(B.T),B.dot(d.T))
		else:
			w = weights.reshape(-1,weights.shape[-1])
			amps = np.zeros([naz+nt,d.shape[0]],dtype=tod.dtype)
			for di in range(len(tod)):
				amps[:,di] = np.linalg.solve(B.dot(w[di,:,None]*B.T),B.dot(w[di]*d[di]))
		#print "amps", amps[:,0]
		# Subtract the best fit
		if asign > 0: d -= amps[:naz].T.dot(B[:naz])
		if tsign > 0: d -= amps[naz:naz+nt].T.dot(B[naz:naz+nt])
		if hsign > 0: d -= amps[naz+nt:naz+nt+nhwp].T.dot(B[naz+nt:naz+nt+nhwp])
	if do_gapfill: gapfiller(d, cuts, inplace=True, overlap=context)
	if deslope: utils.deslope(tod, w=8, inplace=True)
	return d.reshape(tod.shape)

def filter_common_board(tod, dets, layout, name=None):
	mapping = np.zeros(layout.ndet,dtype=int)-1
	mapping[dets] = np.arange(len(dets))
	groups = utils.find_equal_groups(layout.pcb[:,None])
	groups = [mapping[g] for g in groups]
	groups = [g[g>0] for g in groups]
	vs = []
	for gi, group in enumerate(groups):
		if len(group) == 0: continue
		d = tod[group]
		w = 1/(np.mean((d[:,1:]-d[:,:-1])**2,1)/2)
		v = np.sum(w[:,None]*d,0)/np.sum(w)
		if name: vs.append(v)
		tod[group] -= v[None]
	vs = np.array(vs)
	if name:
		with h5py.File("v_"+name+".hdf","w") as hfile:
			hfile["data"] = vs
	return tod
