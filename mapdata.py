import os, sys, zipfile, shutil, io
from pixell import bunch
from contextlib import contextmanager
# Module for encapsulating a (possibly split) map and all the metadata it needs.
# On disk they will be represented as a directory or uncompressed zip files containing:
#  map{i}.fits
#  ivar{i}.fits
#  beam.txt # tfun
#  info.txt # lines of: gain: num, freq: num (GHz)
# These can all be symlinks. It should be easy to convert from the symlink form to the full form,
# so the files can be easily sent elsewhere.
#
# Let's keep heavy imports in the functions that need them, so that this is suitable for building
# command-line tools.

# Functions we want:
# 1. Build symlink or full file from [existing symlink file] + [paths] + [explicit values for info]
# 2. Read symlink or full file into internal representation
# 3. Write full file from internal representation

#######################################################
#### Functions for reading the mapdata into memory ####
#######################################################

def read(fname, splits=None, type="auto", maxmaps=1000, **kwargs):
	"""Read the maps, ivars, beam, gain and fequency from a mapdata file, and
	return them as a bunch of maps:[map,...], ivars:[ivar,...], beam:[:],
	gain:num, freq:num. All maps are read by default. Use the splits argument
	to read only a subset, e.g. splits=[0] to read only the first map for
	each of maps and ivars.

	All unrecognized arguments are forwarded to enmap.read_fits and used when
	reading the maps, allowing for subset reading etc."""
	import numpy as np
	from pixell import enmap
	if   type == "auto": type = infer_type(fname)
	if   type == "zip": work, flexget = zipfile.ZipFile(fname, "r"), zip_flexopen
	elif type == "dir": work, flexget = fname, dir_flexopen
	elif type == "mapinfo": work, flexget = fname, mapinfo_flexget
	else: raise ValueError("Unrecognized type '%s'" % str(type))
	data = bunch.Bunch(maps=[], ivars=[], beam=None, gain=None, freq=None)
	with flexget(work, "info.txt") as f: read_info(f, data)
	with flexget(work, "beam.txt") as f:
		# This supports theformats [l,b,...] and [b]. The beam is assumed to
		# start at l=0 and have a step of 1
		data.beam = read_beam(f)
	if splits is None:
		for i in range(0, maxmaps):
			try:
				with flexget(work, "map%d.fits"  % (i+1)) as f: mapfile  = enmap.read_fits(f, **kwargs)
				with flexget(work, "ivar%d.fits" % (i+1)) as f: ivarfile = enmap.read_fits(f, **kwargs)
			except FileNotFoundError: break
			data["maps"].append(mapfile)
			data["ivars"].append(ivarfile)
	else:
		for i in range(splits):
			with flexget(work, "map%d.fits"  % (i+1)) as f: data["maps"] .append(enmap.read_fits(f, **kwargs))
			with flexget(work, "ivar%d.fits" % (i+1)) as f: data["ivars"].append(enmap.read_fits(f, **kwargs))
	if type == "zip": work.close()
	return data

def read_meta(fname, type="auto", maxmaps=1000, **kwargs):
	"""Read metadata for the given mapdata file. Returns a bunch of
	nmap, map_geometry, ivar_geometry"""
	import numpy as np
	from pixell import enmap
	if   type == "auto": type = infer_type(fname)
	if   type == "zip": work, flexget, has = zipfile.ZipFile(fname, "r"), zip_flexopen, zip_has
	elif type == "dir": work, flexget, has = fname, dir_flexopen, dir_has
	elif type == "mapinfo": work, flexget, has = fname, mapinfo_flexget, mapinfo_has
	else: raise ValueError("Unrecognized type '%s'" % str(type))
	meta = bunch.Bunch(nmap=0, map_geometry=None, ivar_geometry=None)
	with flexget(work, "map1.fits")  as f: meta["map_geometry"]  = enmap.read_fits_geometry(f, **kwargs)
	with flexget(work, "ivar1.fits") as f: meta["ivar_geometry"] = enmap.read_fits_geometry(f, **kwargs)
	for i in range(maxmaps):
		if has(work, "map%d.fits" % (i+1)): meta.nmap = i+1
		else: break
	with flexget(work, "beam.txt") as f: meta.beam = read_beam(f)
	with flexget(work, "info.txt") as f: read_info(f, meta)
	if type == "zip": work.close()
	return meta

def write(fname, data, splits=None, type="auto", maxmaps=1000, **kwargs):
	import numpy as np
	from pixell import enmap, utils
	if   type == "auto": type = infer_type(fname)
	if   type == "zip": work, flexopen = zipfile.ZipFile(fname, "w"), zip_flexopen
	elif type == "dir":
		utils.mkdir(fname)
		work, flexopen = fname, dir_flexopen
	else: raise ValueError("Unrecognized type '%s'" % str(type))
	with flexopen(work, "info.txt", "w") as f: write_info(f, data)
	with flexopen(work, "beam.txt", "w") as f:
		np.savetxt(f, np.array([np.arange(len(data.beam)), data.beam]).T, fmt="%5.0f %15.7e")
	for i, m in enumerate(data.maps):
		with flexopen(work, "map%d.fits"  % (i+1), "w") as f: enmap.write_fits(f, m)
	for i, m in enumerate(data.ivars):
		with flexopen(work, "ivar%d.fits" % (i+1), "w") as f: enmap.write_fits(f, m)

####################################################
#### Functions for manipulating mapdata on disk ####
####################################################

def build_mapinfo(mapfiles=None, ivarfiles=None, beamfile=None, infofile=None,
		gain=None, freq=None, mapdatafile=None):
	"""Build a mapinfo dictionary based on a combination of:
	mapdatafile: Path to an existing symbolic-link mapdata file. Any information not specified in
	  the other arguments will be taken from here
	mapfiles[], ivarfiles[], beamfile, infofile: Paths (or lists of paths for mapfiles or ivarfiles)
	  to the maps, ivars, beam and info that make up the mapdata file.
	gain: real number of the gain correction to use with these maps. Overrides the mapdatafile and infofile values.
	freq: As gain, but for the central frequency in GHz.
	"""
	data = bunch.Bunch(maps=[], ivars=[], beam=None, gain=None, freq=None)
	if mapdatafile is not None: data = read_mapinfo(mapdatafile)
	if mapfiles    is not None: data["maps"]  = [os.path.realpath(fname) for fname in mapfiles]
	if ivarfiles   is not None: data["ivars"] = [os.path.realpath(fname) for fname in ivarfiles]
	if beamfile    is not None: data["beam"]  = os.path.realpath(beamfile)
	if infofile    is not None:
		with open(infofile, "rb") as f:
			read_info(f, data)
	if gain        is not None: data["gain"] = gain
	if freq        is not None: data["freq"] = freq
	if data["gain"] is None: data["gain"] = 1.0
	return data

def read_mapinfo(fname, type="auto", maxmaps=1000):
	"""Reads the filenames (+info) from a link-type mapdata file, returning a dictionary
	{maps:[fname,...], ivars:[fname,...], beam:fname, info:{beam:num, freq:num}}"""
	if type == "auto": type = infer_type(fname)
	res = bunch.Bunch(maps=[], ivars=[], beam=None, gain=None, freq=None)
	if type == "zip":
		with zipfile.ZipFile(fname, "r") as ifile:
			res["beam"] = zip_readlink(ifile, "beam.txt")
			with ifile.open("info.txt", "r") as f: read_info(f, res)
			for i in range(0, maxmaps):
				try:
					mapfile  = zip_readlink(ifile, "map%d.fits"  % (i+1))
					ivarfile = zip_readlink(ifile, "ivar%d.fits" % (i+1))
				except KeyError: break
				res["maps"].append(mapfile)
				res["ivars"].append(ivarfile)
			return res
	elif type == "dir":
		res["beam"] = getlink(fname + "/beam.txt")
		with open(fname + "/info.txt", "rb") as f: read_info(f, res)
		for i in range(0, maxmaps):
			try:
				mapfile  = getlink(fname + "/map%d.fits"  % (i+1))
				ivarfile = getlink(fname + "/ivar%d.fits" % (i+1))
			except FileNotFoundError: break
			res["maps"].append(mapfile)
			res["ivars"].append(ivarfile)
		return res
	else:
		raise ValueError("Unrecognized type '%s'" % str(type))

def write_mapinfo(mapdatafile, mapinfo, mode="link", type="auto"):
	"""Write a mapinfo dictionary mapinfo to the given mapdata file.

	mode: Controls the type of mapdata file written.
	  link: The file will contain (absolute) symbolic links, making it very lightweight, but specific
	    to the system it was made on.
	  copy: The file will contain actual files.
	  The info file will always be a full file, regardless of this argument.

	type: Controls the file type written:
	  "dir": Write a directory with the structure:
	     map{i}.fits
	     ivar{i}.fits
	     beam.txt
	     info.txt
	  "zip": As dir, but as an uncompressed zip file
		"auto" (default): zip if ofilename ends with .zip, otherwise dir
	"""
	if   type == "auto": type = infer_type(mapdatafile)
	rm_r(mapdatafile)
	if   type == "zip":
		with zipfile.ZipFile(mapdatafile, "w") as ofile:
			with ofile.open("info.txt", "w") as f: write_info(f, mapinfo)
			if mode == "link":
				zip_writelink(ofile, "beam.txt", mapinfo["beam"])
				for i, (mfile, ivfile) in enumerate(zip(mapinfo["maps"], mapinfo["ivars"])):
					zip_writelink(ofile, "map%d.fits"  % (i+1), mfile)
					zip_writelink(ofile, "ivar%d.fits" % (i+1), ivfile)
			elif mode == "copy":
				ofile.write(mapinfo["beam"], "beam.txt")
				for i, (mfile, ivfile) in enumerate(zip(mapinfo["maps"], mapinfo["ivars"])):
					ofile.write(mfile , "map%d.fits"  % (i+1))
					ofile.write(ivfile, "ivar%d.fits" % (i+1))
			else: raise ValueError("Unrecognized mode '%s'" % str(mode))
	elif type == "dir":
		os.makedirs(mapdatafile)
		with open(mapdatafile + "/info.txt", "wb") as f: write_info(f, mapinfo)
		if mode == "link":
			os.symlink(mapinfo["beam"], mapdatafile + "/beam.txt")
			for i, (mfile, ivfile) in enumerate(zip(mapinfo["maps"], mapinfo["ivars"])):
				os.symlink(mfile,  mapdatafile + "/map%d.fits"  % (i+1))
				os.symlink(ivfile, mapdatafile + "/ivar%d.fits" % (i+1))
		elif mode == "copy":
			shutil.copyfile(mapinfo["beam"], mapdatafile + "/beam.txt")
			for i, (mfile, ivfile) in enumerate(zip(mapinfo["maps"], mapinfo["ivars"])):
				shutil.copyfile(mfile,  mapdatafile + "/map%d.fits"  % (i+1))
				shutil.copyfile(ivfile, mapdatafile + "/ivar%d.fits" % (i+1))
		else: raise ValueError("Unrecognized mode '%s'" % str(mode))
	else:
		raise ValueError("Unrecognized type '%s'" % str(type))


############################
##### Helper functions #####
############################

def getlink(fname):
	if not os.path.islink(fname): raise IOError("%s is not a symlink" % fname)
	return os.path.realpath(os.readlink(fname))

def read_info(fileobj, out=None):
	if out is None: out = bunch.Bunch()
	try:
		out.beam = fileobj.beam
		out.freq = fileobj.freq
	except AttributeError:
		for line in fileobj:
			line = line.decode()
			toks = line.split(":")
			if len(toks) == 0 or line.startswith("#"): continue
			if   toks[0] == "gain": out["gain"] = float(toks[1])
			elif toks[0] == "freq": out["freq"] = float(toks[1])
			else: raise IOError("Unrecognized key '%s' in info ifle" % (toks[0]))
	return out

def write_info(fileobj, info):
	fileobj.write(("gain:%.8g\nfreq:%.8g\n" % (info["gain"], info["freq"])).encode())

def read_beam(f):
	import numpy as np
	beam = np.loadtxt(f, ndmin=2).T
	return beam[min(1, len(beam)-1)]

def infer_type(fname):
	if isinstance(fname, str):
		if fname.endswith(".zip"): return "zip"
		else:                      return "dir"
	else: return "mapinfo"

def zip_writelink(zfile, name, linkpath):
	zipInfo = zipfile.ZipInfo(name)
	zipInfo.external_attr |= 0xA0000000;
	zfile.writestr(zipInfo, linkpath)

def zip_readlink(zfile, name):
	return zfile.read(name).decode() # this assumes unicode paths, might change

@contextmanager
def zip_flexopen(zfile, name, mode="r"):
	"""Open name in zipfile, returning a file-like object unless it's
	a symlink, in which case the path string is returned"""
	try:
		obj  = None
		if mode == "r":
			# When reading, support symlinks
			info = zfile.getinfo(name)
			link = info.external_attr & 0xA0000000 == 0xA0000000
			if link: yield zfile.read(name).decode()
			else:
				obj = zfile.open(name, "r")
				yield obj
		else:
			obj = zfile.open(name, "w")
			with two_step_write(obj) as f:
				yield f
	except KeyError: raise FileNotFoundError
	finally:
		if obj: obj.close()

@contextmanager
def dir_flexopen(dirpath, name, mode="r"):
	try:
		obj   = None
		fname = dirpath + "/" + name
		if mode == "r":
			# When reading, support symlinks
			link  = os.path.islink(fname)
			if link: yield getlink(fname)
			else:
				obj = open(fname, "rb")
				yield(obj)
		else:
			obj = open(fname, mode+"b")
			yield obj
	finally:
		if obj: obj.close()

@contextmanager
def mapinfo_flexget(dirpath, name):
	# This is pretty hacky, but lets us treat a mapinfo object the same way as the other formats
	try:
		pre = name.split(".")[0]
		if   pre == "info": yield dirpath
		elif pre == "beam": yield dirpath["beam"]
		elif pre.startswith("map"):
			i = int(pre[3:])
			if i >= len(dirpath.maps): raise FileNotFoundError
			else: yield dirpath.maps[i]
		elif pre.startswith("ivar"):
			i = int(pre[4:])
			if i >= len(dirpath.ivars): raise FileNotFoundError
			else: yield dirpath.ivars[i]
		else: raise FileNotFoundError
	finally:
		pass

@contextmanager
def two_step_write(f):
	try:
		buf = io.BytesIO()
		yield buf
	finally:
		f.write(buf.getvalue())

def zip_has(zfile, name):
	try:
		zfile.getinfo(name)
		return True
	except KeyError:
		return False

def dir_has(dirpath, name): return os.path.isfile(dirpath + "/" + name)
def mapinfo_has(dirpath, name):
	try:
		with mapinfo_flexget(dirpath, name) as f: pass
		return True
	except FileNotFoundError:
		return False

# Why is python so stupid here?
def rm_r(path):
	try: shutil.rmtree(path)
	except FileNotFoundError: pass
	except NotADirectoryError:
		os.remove(path)
