# Transform from real numbers to RGB colors.
import numpy as np, time
import fortran

# Predefined schemes
schemes = {}

class Colorscheme:
	def __init__(self, desc):
		"""Parses a color description string of the form "v1:c1,v2:c2,...,vn,vn"
		into a numpy array of values [v1,v2,..,vn] and a numpy array of colors,
		[[r,g,b,a],[r,g,b,a],[r,g,b,a],...]."""
		try:
			desc = schemes[desc]
		except KeyError:
			pass
		try:
			self.vals, self.cols, self.desc = desc.vals, desc.cols, desc.desc
			return
		except AttributeError:
			pass
		toks = desc.split(",")
		# Construct the output arrays
		vals = np.zeros((len(toks)))
		cols = np.zeros((len(toks),4))
		# And populate them
		for i, tok in enumerate(toks):
			val, code = tok.split(":")
			vals[i] = float(val)
			color = np.array((0,0,0,0xff),dtype=np.uint8)
			m = len(code)/2
			for j in range(m):
				color[j] = int(code[2*j:2*(j+1)],16)
			cols[i,:] = color
		# Sort result
		order = np.argsort(vals)
		self.vals, self.cols = vals[order], cols[order]
		self.desc = desc

schemes["wmap"]    = Colorscheme("0:000080,0.15:0000ff,0.4:00ffff,0.7:ffff00,0.9:ff5500,1:800000")
schemes["gray"]    = Colorscheme("0:000000,1:ffffff")
schemes["hotcold"] = Colorscheme("0:0000ff,0.5:000000,1:ff0000")
schemes["cooltowarm"] = Colorscheme("0.00000:3b4cc0,0.03125:445acc,0.06250:4d68d7,0.09375:5775e1,0.12500:6282ea,0.15625:6c8ef1,0.18750:779af7,0.21875:82a5fb,0.25000:8db0fe,0.28125:98b9ff,0.31250:a3c2ff,0.34375:aec9fd,0.37500:b8d0f9,0.40625:c2d5f4,0.43750:ccd9ee,0.46875:d5dbe6,0.50000:dddddd,0.53125:e5d8d1,0.56250:ecd3c5,0.59375:f1ccb9,0.62500:f5c4ad,0.65625:f7bba0,0.68750:f7b194,0.71875:f7a687,0.75000:f49a7b,0.78125:f18d6f,0.81250:ec7f63,0.84375:e57058,0.87500:de604d,0.90625:d55042,0.93750:cb3e38,0.96875:c0282f,1.00000:b40426")

def colorize(arr, desc="wmap", method="simple"):
	# Accept both color schemes and strings
	desc = Colorscheme(desc)
	if len(desc.vals) == 0:
		return np.zeros(arr.shape+(4,),dtype=np.uint8)
	elif len(desc.vals) == 1:
		return np.tile(desc.cols[0],arr.shape+(1,)).T
	else:
		a   = arr.reshape(-1)
		if method == "simple":
			res = colorize_simple(a, desc)
		elif method == "fast":
			res = colorize_fast(a, desc)
		else:
			raise NotImplementedError("colorize method '%d' is not implemented" % method)
		return res.reshape(arr.shape+(4,))

def colorize_fast(a, desc):
	res = np.empty((len(a),4),dtype=np.uint16)
	fortran.remap(a, res.T, desc.vals, desc.cols.astype(np.int16).T)
	return res.astype(np.uint8)

def colorize_simple(a, desc):
	res = np.empty((len(a),4),dtype=np.uint8)
	ok  = np.where(~np.isnan(a))
	bad = np.where( np.isnan(a))
	# Bad values are transparent
	res[bad,:] = np.array((0,0,0,0),np.uint8)
	# Good ones get proper treatment
	i = np.searchsorted(desc.vals, a[ok])
	# We always want a point to our left and right
	i = np.minimum(np.maximum(i,1),len(desc.vals)-1)
	# Fractional distance to next point
	x = (a[ok] - desc.vals[i-1])/(desc.vals[i]-desc.vals[i-1])
	# Cap this value too
	x = np.minimum(np.maximum(x,0),1)
	# The result is the linear combination of the two
	# end points
	col = np.round(desc.cols[i-1]*(1-x)[:,None] + desc.cols[i]*x[:,None])
	res[ok] = np.array(np.minimum(np.maximum(col,0),0xff),dtype=np.uint8)
	return res
