# This is a much simpler implementation of mapdata that will probably take over
# for it. The format is simply a text file with the format
# map  path
# ivar path
# [map2 path]
# [ivar2 path]
# [....]
# beam path_to_beam_transform
# freq freq_in_GHz
import numpy as np, os
from pixell import bunch, enmap, utils

def read(fname, splits=None, maxmaps=1000, output="mvb", **kwargs):
	"""Read the maps, ivars, beam, gain and fequency from a mapdata file, and
	return them as a bunch of maps:[map,...], ivars:[ivar,...], beam:[:],
	gain:num, freq:num. All maps are read by default. Use the splits argument
	to read only a subset, e.g. splits=[0] to read only the first map for
	each of maps and ivars.

	All unrecognized arguments are forwarded to enmap.read_fits and used when
	reading the maps, allowing for subset reading etc."""
	paths = read_info(fname)
	for key in ["map","ivar","beam","freq"]:
		if key not in paths:
			raise ValueError("Expected key '%s' not found in mapdata file" % key)
	if splits is None:
		splits = []
		for i in range(maxmaps):
			tag = "" if i == 0 else str(i+1)
			if "map"+tag in paths and "ivar"+tag in paths:
				splits.append(i)
			else:
				break
	freq = float(paths.freq)
	# Read the beam, getting only the b(l) part
	res  = bunch.Bunch(freq=freq, beam=None, maps=[], ivars=[], raw=paths.raw)
	if "b" in output: res.beam = get_beam(paths.beam)
	for i in splits:
		tag = "" if i == 0 else str(i+1)
		if "m" in output:
			res.maps .append(enmap.read_map(paths["map" +tag], **kwargs))
		if "v" in output:
			res.ivars.append(enmap.read_map(paths["ivar"+tag], **kwargs))
	return res

def read_meta(fname):
	"""Read metadata for the given mapdata file. Returns a bunch of
	nmap, map_geometry, ivar_geometry"""
	paths = read_info(fname)
	for key in ["map","ivar","beam","freq"]:
		if key not in paths:
			raise ValueError("Expected key '%s' not found in mapdata file" % key)
	nmap = 1
	while "map%d" % nmap in paths:
		nmap += 1
	res = bunch.Bunch(
			nmap = nmap,
			map_geometry  = enmap.read_map_geometry(paths.map),
			ivar_geometry = enmap.read_map_geometry(paths.ivar),
			raw = paths.raw,
		)
	return res

def read_info(fname):
	"""Helper function. Reads key-value pairs from text file
	and returns bunch of them"""
	res = bunch.Bunch()
	raw = bunch.Bunch()
	with open(fname, "r") as f:
		for line in f:
			line = line.strip()
			# Skip empty lines and comments
			if len(line) == 0:       continue
			if line.startswith("#"): continue
			toks = utils.split_outside(line, sep=" ", start="\"'", end="\"'")
			toks = [tok for tok in toks if len(tok) > 0 ]
			if len(toks) != 2:
				raise ValueError("Error parsing key-value file. Expected format key value, but got '%s'" % (line))
			# Expand relative paths
			key, val = toks
			raw[key] = val
			if not is_num(val):
				val = os.path.join(os.path.dirname(fname), val)
			res[key] = val
	res.raw = raw
	return res

def get_beam(fname_or_fwhm, lmax=40000):
	try:
		sigma = float(fname_or_fwhm)*utils.arcmin*utils.fwhm
		l     = np.arange(lmax+1)
		return np.exp(-0.5*l**2*sigma**2)
	except ValueError:
		# Read the beam, getting only the b(l) part
		beam = np.loadtxt(fname_or_fwhm, ndmin=2).T
		beam = beam[min(1, len(beam)-1)]
		return beam

def is_num(s):
	try:
		float(s)
		return True
	except ValueError:
		return False
