"""This module provides fast weightmap estimation based on a todinfo database."""
import numpy as np, os
from enlib import utils, coordinates, enmap, bench, pmat

def fastweight(shape, wcs, db, weight="det", array_rad=0.7*utils.degree,
		comm=None, dtype=np.float64, daz=0.5*utils.degree, nt=4, chunk_size=100,
		site=None, verbose=False, normalize=True):
	# Get the boresight bounds for each TOD
	ntod   = len(db)
	mids   = np.array([db.data["t"],db.data["az"],db.data["el"]])
	widths = np.array([db.data["dur"],db.data["waz"],db.data["wel"]])
	box    = np.array([mids-widths/2,mids+widths/2])
	box[:,1:] *= utils.degree
	ndets  = db.data["ndet"]
	# Set up our output map
	omap = enmap.zeros(shape, wcs, dtype)
	# Sky horizontal period in pixels
	nphi = np.abs(utils.nint(360/wcs.wcs.cdelt[0]))
	# Loop through chunks
	nchunk= (ntod+chunk_size-1)/chunk_size
	if comm: rank, size = comm.rank, comm.size
	else:    rank, size = 0, 1
	for chunk in range(rank, nchunk, size):
		i1 = chunk*chunk_size
		i2 = min((chunk+1)*chunk_size, ntod)
		# Split the hits into horizontal pixel ranges
		pix_ranges, weights = [], []
		with bench.mark("get"):
			for i in range(i1,i2):
				ndet_eff = ndets[i] if weight == "det" else 1000.0
				pr, w = get_pix_ranges(shape, wcs, box[:,:,i], daz, nt, ndet=ndet_eff, site=site)
				if pr is None: continue
				pix_ranges.append(pr)
				weights.append(w)
			if len(pix_ranges) == 0: continue
			pix_ranges = np.concatenate(pix_ranges, 0)
			weights    = np.concatenate(weights, 0)
		with bench.mark("add"):
			add_weight(omap, pix_ranges, weights, nphi)
		if verbose:
			print "%4d %4d %7.4f %7.4f" % (chunk, comm.rank, bench.stats.get("get"), bench.stats.get("add"))
	if comm:
		omap = utils.allreduce(omap, comm)
	# Change unit from seconds per pixel to seconds per square acmin
	if normalize:
		pixarea = omap.pixsizemap() / utils.arcmin**2
		omap   /= pixarea
	omap[~np.isfinite(omap)] = 0
	if array_rad:
		omap = smooth_tophat(omap, array_rad)
	omap[omap<1e-6] = 0
	return omap

def fixx(yx, nphi):
	yx    = np.array(yx)
	yx[1] = utils.unwind(yx[1], nphi)
	return yx

def get_pix_ranges(shape, wcs, horbox, daz, nt=4, ndet=1.0, site=None):
	"""An appropriate daz for this function is about 1 degree"""
	# For each row in the map we want to know the hit density for that row,
	# as well as its start and end. In the original function we got one
	# sample per row by oversampling and then using unique. This is unreliable,
	# and also results in quantized steps in the depth. We can instead
	# do a coarse equispaced az -> ra,dec -> y,x. We can then interpolate
	# this to get exactly one sample per y. To get the density properly,
	# we just need dy/dt = dy/daz * daz/dt, where we assume daz/dt is constant.
	# We get dy/daz from the coarse stuff, and interpolate that too, which gives
	# the density per row.
	(t1,t2),(az1,az2),el = horbox[:,0], horbox[:,1], np.mean(horbox[:,2])
	nphi = np.abs(utils.nint(360/wcs.wcs.cdelt[0]))
	# First produce the coarse single scan
	naz  = utils.nint(np.abs(az2-az1)/daz)
	if naz <= 1: return None, None
	ahor = np.zeros([3,naz])
	ahor[0] = utils.ctime2mjd(t1)
	ahor[1] = np.linspace(az1,az2,naz)
	ahor[2] = el
	acel    = coordinates.transform("hor","cel",ahor[1:],time=ahor[0],site=site)
	ylow, x1low = fixx(enmap.sky2pix(shape, wcs, acel[::-1]),nphi)
	if ylow[1] < ylow[0]:
		ylow, x1low = ylow[::-1], x1low[::-1]
	# Find dy/daz for these points
	glow   = np.gradient(ylow)*(naz-1)/(az2-az1)
	# Now interpolate to full resolution
	y    = np.arange(utils.nint(ylow[0]),utils.nint(ylow[-1])+1)
	if len(y) == 0:
		print "Why is y empty?", naz, ylow[0], ylow[1]
		return None, None
	x1   = np.interp(y, ylow, x1low)
	grad = np.interp(y, ylow, glow)
	# Now we just need the width of the rows, x2, which comes
	# from the time drift
	thor = np.zeros([3,nt])
	thor[0] = utils.ctime2mjd(np.linspace(t1,t2,nt))
	thor[1] = az1
	thor[2] = el
	tcel    = coordinates.transform("hor","cel",thor[1:],time=thor[0],site=site)
	_, tx   = utils.nint(fixx(enmap.sky2pix(shape, wcs, tcel[::-1]),nphi))
	x2 = x1 + tx[-1]-tx[0]
	x1, x2  = np.minimum(x1,x2), np.maximum(x1,x2)
	pix_ranges = utils.nint(np.concatenate([y[:,None],x1[:,None],x2[:,None]],1))
	# Weight per pixel. We want this to be in units of seconds of
	# observing time per pixel if ndet=1. We know the total number of pixels
	# hit (ny*nx) and the total time (t2-t1), and we know the relative
	# weight per row (1/grad), so we can just normalize things
	ny, nx = len(y), x2[0]-x1[0]
	npix = ny*nx
	if npix == 0 or np.any(grad <= 0):
		return pix_ranges, grad*0
	else:
		weights = 1/grad
		weights *= (t2-t1)/(np.sum(weights)*nx) * ndet # *nx because weight is per row
		return pix_ranges, weights

def add_weight(omap, pix_ranges, weights, nphi=0, method="fortran"):
	if   method == "fortran": add_weight_fortran(omap, pix_ranges, weights, nphi)
	elif method == "python":  add_weight_python (omap, pix_ranges, weights, nphi)
	else: raise ValueError
def add_weight_python(omap, pix_ranges, weights, nphi=0):
	# This function is a candidate for implementation in fortran
	for (y,x1,x2), w in zip(pix_ranges, weights):
		omap[y,max(0,x1):min(omap.shape[1],x2)] += w
def add_weight_fortran(omap, pix_ranges, weights, nphi=0):
	core = pmat.get_core(omap.dtype)
	core.add_rows(omap.T, pix_ranges[:,0], pix_ranges[:,1:].T, weights, nphi)

enmap.extent_model = ["intermediate"]
def smooth_tophat(map, rad):
	# Will use flat sky approximation here. It's not a good approximation for
	# our big maps, but this doesn't need to be accurate anyway
	ny,nx = map.shape[-2:]
	refy, refx = ny/2,nx/2
	pos   = map.posmap()
	pos[0] -= pos[0,refy,refx]
	pos[1] -= pos[1,refy,refx]
	r2     = np.sum(pos**2,0)
	kernel  = (r2 < rad**2).astype(map.dtype)
	kernel /= np.sum(kernel)
	kernel *= map.size**0.5
	kernel = np.roll(kernel,-refy,0)
	kernel = np.roll(kernel,-refx,1)
	res = enmap.ifft(enmap.fft(map)*np.conj(enmap.fft(kernel))).real
	return res
