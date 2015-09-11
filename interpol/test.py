import numpy as np, time
from enlib.interpol import spline_filter, map_coordinates
from scipy import ndimage

def spline_filter_mat(n, order=3, border="mirror", trans=False, scipy=False):
	out = np.zeros((n,n))
	for i in range(n):
		data = np.zeros(n)
		data[i] = 1
		if not scipy:
			out[i] = spline_filter(data, order=order, border=border, trans=trans)
		else:
			if order > 1:
				out[i] = ndimage.spline_filter(data, order=order)
	return out

def interpol_mat(n, m, mode="spline", order=3, border="mirror", trans=False, scipy=False):
	idata = np.zeros(m)
	odata = np.zeros(n)
	point = np.linspace(0, m, n, endpoint=False)[None]
	if not trans:
		out = np.zeros((m,n))
		for i in range(m):
			idata[:] = 0
			idata[i] = 1
			if not scipy:
				map_coordinates(idata, point, odata, mode=mode, order=order, border=border, trans=trans)
			else:
				if border == "mirror": sborder = "reflect"
				elif border == "cyclic": sborder = "wrap"
				else: sborder = "constant"
				odata[:] = ndimage.map_coordinates(idata, point, order=order, mode=sborder)
			out[i] = odata
		return out
	else:
		out = np.zeros((n,m))
		for i in range(n):
			odata[:] = 0
			odata[i] = 1
			if not scipy:
				map_coordinates(idata, point, odata, mode=mode, order=order, border=border, trans=trans)
			else:
				idata[:] = 0
			out[i] = idata
		return out

def dstat(a): return np.std(a)

# Test spline filter for symmetry and ndimage equivalence. Should only
# match ndimage for mirror boundary conditions.
print "Testing spline filter"
n = 10
for border in ["zero", "nearest", "cyclic", "mirror"]:
	for order in range(6):
#for border in ["mirror"]:
#	for order in [3]:
		t = [time.time()]
		Mf = spline_filter_mat(n, order=order, border=border, trans=False)
		t.append(time.time())
		MfT= spline_filter_mat(n, order=order, border=border, trans=True)
		t.append(time.time())
		Mf2= spline_filter_mat(n, order=order, border=border, trans=False, scipy=True)
		t.append(time.time())
		#print "Mf"
		#print Mf
		#print "Mf2"
		#print Mf2
		#print "MfT.T"
		#print MfT.T
		#print "Mf-MfT.T"
		#print Mf-MfT.T
		print "%-10s %d %15.7e %15.7e %7.4f %7.4f %7.4f" % (border, order, dstat(Mf-MfT.T), dstat(Mf-Mf2), t[1]-t[0], t[2]-t[1], t[3]-t[2])

# Test coordinate mapping for symmetry and ndimage equivalence.
print "Testing interpol"
n = 300; m = 100
for mode in ["conv", "spline", "lanczos"]:
	for border in ["zero", "nearest", "cyclic", "mirror"]:
		for order in range(6):
			t = [time.time()]
			Mf = interpol_mat(n, m, order=order, mode=mode, border=border, trans=False)
			t.append(time.time())
			MfT= interpol_mat(n, m, order=order, mode=mode, border=border, trans=True)
			t.append(time.time())
			Mf2= interpol_mat(n, m, order=order, mode=mode, border=border, trans=False, scipy=True)
			t.append(time.time())
			#print "Mf"
			#print Mf
			#print "Mf2"
			#print Mf2
			#print "MfT.T"
			#print MfT.T
			#print "Mf-MfT.T"
			#print Mf-MfT.T
			print "%-10s %-10s %d %15.7e %15.7e %7.4f %7.4f %7.4f" % (mode, border, order, dstat(Mf-MfT.T), dstat(Mf-Mf2), t[1]-t[0], t[2]-t[1], t[3]-t[2])
