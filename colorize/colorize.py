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
		if len(toks) == 1:
			# Constant color mode
			desc = "0:%s,1:%s" % (desc,desc)
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

schemes["gray"]    = Colorscheme("0:000000,1:ffffff")
schemes["wmap"]    = Colorscheme("0:000080,0.15:0000ff,0.4:00ffff,0.7:ffff00,0.9:ff5500,1:800000")
schemes["planck"]  = Colorscheme("0:0000ff,0.33:ffedd9,0.83:ff4b00,1:640000")
schemes["hotcold"] = Colorscheme("0:0000ff,0.5:000000,1:ff0000")
schemes["cooltowarm"] = Colorscheme("0.00000:3b4cc0,0.03125:445acc,0.06250:4d68d7,0.09375:5775e1,0.12500:6282ea,0.15625:6c8ef1,0.18750:779af7,0.21875:82a5fb,0.25000:8db0fe,0.28125:98b9ff,0.31250:a3c2ff,0.34375:aec9fd,0.37500:b8d0f9,0.40625:c2d5f4,0.43750:ccd9ee,0.46875:d5dbe6,0.50000:dddddd,0.53125:e5d8d1,0.56250:ecd3c5,0.59375:f1ccb9,0.62500:f5c4ad,0.65625:f7bba0,0.68750:f7b194,0.71875:f7a687,0.75000:f49a7b,0.78125:f18d6f,0.81250:ec7f63,0.84375:e57058,0.87500:de604d,0.90625:d55042,0.93750:cb3e38,0.96875:c0282f,1.00000:b40426")
schemes["cubehelix"] = Colorscheme("0.000:000000,0.004:020002,0.008:040103,0.012:060105,0.016:080207,0.020:0a0209,0.024:0b020b,0.027:0d030d,0.031:0e030f,0.035:100412,0.039:110514,0.043:120516,0.047:140618,0.051:15071b,0.055:16071d,0.059:16081f,0.063:170922,0.067:180a24,0.071:190b26,0.075:190c29,0.078:1a0d2b,0.082:1a0e2d,0.086:1a0f2f,0.090:1a1032,0.094:1a1234,0.098:1a1336,0.102:1a1438,0.106:1a163a,0.110:1a173c,0.114:1a193e,0.118:191a40,0.122:191c42,0.125:181d44,0.129:181f46,0.133:172147,0.137:172249,0.141:16244b,0.145:15264c,0.149:14284d,0.153:142a4f,0.157:132b50,0.161:122d51,0.165:112f52,0.169:103153,0.173:0f3354,0.176:0e3554,0.180:0d3755,0.184:0c3955,0.188:0c3c56,0.192:0b3e56,0.196:0a4056,0.200:094256,0.204:084457,0.208:074656,0.212:074856,0.216:064a56,0.220:054c56,0.224:054e55,0.227:045055,0.231:045254,0.235:035453,0.239:035652,0.243:035852,0.247:035a51,0.251:025c50,0.255:025e4e,0.259:03604d,0.263:03624c,0.267:03644b,0.271:036549,0.275:046748,0.278:046947,0.282:056a45,0.286:066c44,0.290:076d42,0.294:086f40,0.298:09703f,0.302:0a723d,0.306:0b733c,0.310:0d743a,0.314:0e7538,0.318:107737,0.322:127835,0.325:147933,0.329:167a32,0.333:187b30,0.337:1a7b2f,0.341:1d7c2d,0.345:1f7d2c,0.349:227e2a,0.353:247e29,0.357:277f27,0.361:2a7f26,0.365:2d8025,0.369:308024,0.373:338023,0.376:368122,0.380:398121,0.384:3d8120,0.388:40811f,0.392:44811e,0.396:47811e,0.400:4b811d,0.404:4f811d,0.408:53811d,0.412:56801c,0.416:5a801c,0.420:5e801c,0.424:62801d,0.427:667f1d,0.431:6a7f1d,0.435:6e7e1e,0.439:727e1e,0.443:767d1f,0.447:7a7d20,0.451:7e7c21,0.455:827c22,0.459:867b23,0.463:8a7b25,0.467:8e7a26,0.471:927928,0.475:96792a,0.478:9a782b,0.482:9e782d,0.486:a27730,0.490:a57632,0.494:a97634,0.498:ad7536,0.502:b07539,0.506:b4743c,0.510:b7743e,0.514:ba7341,0.518:bd7344,0.522:c17247,0.525:c4724a,0.529:c7714e,0.533:c97151,0.537:cc7154,0.541:cf7058,0.545:d1705b,0.549:d4705f,0.553:d67062,0.557:d87066,0.561:da706a,0.565:dc706d,0.569:de7071,0.573:e07075,0.576:e27079,0.580:e3707d,0.584:e47181,0.588:e67185,0.592:e77189,0.596:e8728d,0.600:e97291,0.604:ea7395,0.608:ea7399,0.612:eb749c,0.616:eb75a0,0.620:eb76a4,0.624:ec77a8,0.627:ec78ac,0.631:ec79b0,0.635:ec7ab3,0.639:eb7bb7,0.643:eb7cbb,0.647:eb7dbe,0.651:ea7fc2,0.655:e980c5,0.659:e981c8,0.663:e883cc,0.667:e784cf,0.671:e686d2,0.675:e588d5,0.678:e489d8,0.682:e38bdb,0.686:e28ddd,0.690:e08fe0,0.694:df91e3,0.698:de93e5,0.702:dd94e7,0.706:db96ea,0.710:da98ec,0.714:d89aee,0.718:d79df0,0.722:d59ff1,0.725:d4a1f3,0.729:d2a3f5,0.733:d1a5f6,0.737:cfa7f8,0.741:cea9f9,0.745:cdacfa,0.749:cbaefb,0.753:cab0fc,0.757:c8b2fd,0.761:c7b5fe,0.765:c6b7fe,0.769:c5b9ff,0.773:c4bbff,0.776:c2bdff,0.780:c1bfff,0.784:c0c2ff,0.788:bfc4ff,0.792:bfc6ff,0.796:bec8ff,0.800:bdcaff,0.804:bcccff,0.808:bcceff,0.812:bbd0ff,0.816:bbd2ff,0.820:bbd4fe,0.824:bbd6fe,0.827:bad8fd,0.831:bad9fd,0.835:bbdbfc,0.839:bbddfb,0.843:bbdffb,0.847:bbe0fa,0.851:bce2f9,0.855:bce3f9,0.859:bde5f8,0.863:bee6f7,0.867:bfe8f7,0.871:c0e9f6,0.875:c1eaf5,0.878:c2ecf5,0.882:c3edf4,0.886:c4eef3,0.890:c6eff3,0.894:c7f0f2,0.898:c9f1f2,0.902:caf2f1,0.906:ccf3f1,0.910:cef4f1,0.914:cff5f0,0.918:d1f5f0,0.922:d3f6f0,0.925:d5f7f0,0.929:d7f7f0,0.933:d9f8f0,0.937:dbf9f0,0.941:def9f1,0.945:e0faf1,0.949:e2faf1,0.953:e4fbf2,0.957:e7fbf2,0.961:e9fcf3,0.965:ebfcf4,0.969:edfcf5,0.973:f0fdf6,0.976:f2fdf7,0.980:f4fdf8,0.984:f6fef9,0.988:f9fefa,0.992:fbfefc,0.996:fdfffd,1.000:ffffff")
schemes["nozero"]    = Colorscheme("0:000080,0.15:0000ff,0.499998:55ffaa,0.499999:55ffaa00,0.500001:55ffaa00,0.500002:55ffaa,0.4:00ffff,0.7:ffff00,0.9:ff5500,1:800000")

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
		elif method == "direct":
			a = arr.reshape(arr.shape[0],-1)
			res = colorize_direct(a, desc)
			return res.reshape(arr.shape[1:]+(4,))
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

def colorize_direct(a, desc):
	res = np.empty((a.shape[1],4),dtype=np.uint16)
	fortran.direct(a.T, res.T)
	return res.astype(np.uint8)
