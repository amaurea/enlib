import numpy as np, time, argparse
from enlib import enmap, utils
import cy_parallax

parser = argparse.ArgumentParser()
parser.add_argument("--tmax", type=float, default=0.10)
args = parser.parse_args()

box = np.array([[-1,-1],[1,1]])*5*utils.degree
shape, wcs = enmap.geometry(box, res=0.2*utils.arcmin, proj="car", ref=(0,0))
dtype = np.float32
nloop = 1000

imap = enmap.zeros(shape, wcs, dtype)+1
omap = imap*0

times = []
for method in [0,1]:
	t1 = time.time()
	n  = 0
	for i in range(nloop):
		cy_parallax.displace_map(imap, [1,0,0], 500, [1*utils.arcmin, 0], omap, method=method)
		n += 1
		t2 = time.time()
		if t2-t1 > args.tmax: break
	t = (t2-t1)/n
	times.append(t)
	print("%2d %8.3f ms %8.3f x" % (method, t*1e3, times[0]/t))
