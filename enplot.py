import numpy as np, argparse, time, sys, warnings, os, shlex, glob, PIL.Image
from enlib import enmap, colorize, mpi, cgrid, utils

class Printer:
	def __init__(self, level=1, prefix=""):
		self.level  = level
		self.prefix = prefix
	def write(self, desc, level, exact=False, newline=True, prepend=""):
		if level == self.level or not exact and level <= self.level:
			sys.stderr.write(prepend + self.prefix + desc + ("\n" if newline else ""))
	def push(self, desc):
		return Printer(self.level, self.prefix + desc)
	def time(self, desc, level, exact=False, newline=True):
		class PrintTimer:
			def __init__(self, printer): self.printer = printer
			def __enter__(self): self.time = time.time()
			def __exit__(self, type, value, traceback):
				self.printer.write(desc, level, exact=exact, newline=newline, prepend="%6.2f " % (time.time()-self.time))
		return PrintTimer(self)
noprint = Printer(0)

def plot(ifiles, args=None, comm=None, noglob=False):
	"""Plot the given maps, writing each field to disk with the same
	name as each input file, with the extension changed to .png.
	There are two main formats: plot(command_string) and plot(ifiles, args).
	Command string is a command-line argument like what the enplot program
	accepts, and is translated to ifiles, args internally. Args is a
	bunch-like object that can be used to modify how the plots are generated
	and where they are written. See the enplot.parse_args function for a list."""
	if args is None:
		args = parse_args(ifiles, noglob=noglob)
		ifiles = args.ifiles
	if comm is None:
		comm = mpi.COMM_WORLD
	# Set up verbose output
	printer = Printer(args.verbosity)
	# Plot each file
	for fi in range(comm.rank,len(ifiles),comm.size):
		ifile = ifiles[fi]
		with printer.time("read %s" % ifile, 3):
			map   = get_map(ifile, args)
		with printer.time("ranges", 3):
			crange= get_color_range(map, args)
		for ci, cr in enumerate(crange.T):
			printer.write("color range %d: %12.5e to %15.7e" % (ci, cr[0], cr[1]), 4)
		# Loop over map fields
		ncomp  = map.shape[0]
		ngroup = 3 if args.rgb else 1
		for i in range(0, ncomp, ngroup):
			# Construct default out format
			ndigit   = get_num_digits(ncomp)
			subprint = printer.push(("%%0%dd/%%d " % ndigit) % (i+1,ncomp))
			dir, base, ext = split_file_name(ifile)
			map_field = map[i:i+ngroup]
			# Build output file name
			oinfo = {"dir":"" if dir == "." else dir + "/", "base":base, "iext":ext,
					"fi":fi, "fn":len(args.ifiles), "ci":i, "cn":ncomp, "pi":comm.rank, "pn":comm.size,
					"pre":args.prefix, "suf":args.suffix, "comp": "_%0*d" % (ndigit,i) if map.ndim > 2 else "",
					"ext":args.ext, "layer":""}
			oname = args.oname.format(**oinfo)
			# Draw the map
			if args.driver.lower() == "pil":
				img, info = draw_map_field(map_field, args, crange, return_info=True, return_layers=args.layers, printer=subprint)
				padding = np.array([-info.bounds[0,::-1],info.bounds[1,::-1]-map_field.shape[-2:]],dtype=int)
				printer.write("padded by %d %d %d %d" % tuple(padding.reshape(-1)), 4)
				if args.layers:
					for layer, name in zip(img, info.names):
						oinfo["layer"] = "_" + name
						oname = args.oname.format(**oinfo)
						with subprint.time("write to %s" % oname, 3):
							layer.save(oname)
				else:
					with subprint.time("write to %s" % oname, 3):
						img.save(oname)
			elif args.driver.lower() in ["matplotlib","mpl"]:
				figure = draw_map_field_mpl(map_field, args, crange, printer=subprint)
				with subprint.time("write to %s" % oname, 3):
					figure.savefig(oname,bbox_inches="tight",dpi=args.mpl_dpi)
			# Progress report
			printer.write("\r%s %5d/%d" % (ifile, i+1,ncomp), 2, exact=True, newline=False)
		printer.write("",    2, exact=True)
		printer.write(ifile, 1, exact=True)

def parse_args(args=sys.argv[1:], noglob=False):
	parser = argparse.ArgumentParser()
	parser.add_argument("ifiles", nargs="+")
	parser.add_argument("-o", "--oname", default="{dir}{pre}{base}{suf}{comp}{layer}.{ext}")
	parser.add_argument("-c", "--color", default="wmap")
	parser.add_argument("-r", "--range", type=str)
	parser.add_argument("--min", type=str)
	parser.add_argument("--max", type=str)
	parser.add_argument("-q", "--quantile", type=float, default=0.01)
	parser.add_argument("-v", dest="verbosity", action="count")
	parser.add_argument("-s", "-u", "--scale", "--upgrade", type=str, default="1")
	parser.add_argument("--verbosity", dest="verbosity", type=int)
	parser.add_argument("--method", default="fast")
	parser.add_argument("--slice", type=str)
	parser.add_argument("--sub",   type=str)
	parser.add_argument("--op", type=str)
	parser.add_argument("-d", "--downgrade", type=str, default="1")
	parser.add_argument("--prefix", type=str, default="")
	parser.add_argument("--suffix", type=str, default="")
	parser.add_argument("--ext", type=str, default="png")
	parser.add_argument("-m", "--mask", type=float)
	parser.add_argument("--mask-tol", type=float, default=1e-14)
	parser.add_argument("-g", "--grid", action="count", default=1)
	parser.add_argument("--grid-color", type=str, default="00000020")
	parser.add_argument("-t", "--ticks", type=str, default="1")
	parser.add_argument("--nolabels", action="store_true")
	parser.add_argument("--nstep", type=int, default=200)
	parser.add_argument("--subticks", type=float, default=0)
	parser.add_argument("--font", type=str, default="arial.ttf")
	parser.add_argument("--font-size", type=int, default=20)
	parser.add_argument("--font-color", type=str, default="000000")
	parser.add_argument("-D", "--driver", type=str, default="pil")
	parser.add_argument("--mpl-dpi", type=float, default=75)
	parser.add_argument("--mpl-pad", type=float, default=1.6)
	parser.add_argument("--rgb", action="store_true")
	parser.add_argument("-a", "--autocrop", action="store_true")
	parser.add_argument("-A", "--autocrop-each", action="store_true")
	parser.add_argument("-L", "--layers", action="store_true")
	if isinstance(args, basestring):
		oargs = []
		for tok in shlex.split(args):
			if not noglob:
				gtok = glob.glob(tok)
				if len(gtok) == 0: gtok = [tok]
				tok = gtok
			else: tok = [tok]
			oargs += tok
		args = oargs
	return parser.parse_args(args)

def get_map(ifile, args):
	"""Read the specified map, and massage it according to the options
	in args. Relevant ones are sub, autocrop, slice, op, downgrade, scale,
	mask. Retuns with shape [:,ny,nx], where any extra dimensions have been
	flattened into a single one."""
	toks = ifile.split(":")
	ifile, slice = toks[0], ":".join(toks[1:])
	m0 = enmap.read_map(ifile)
	# Save the original map, so we can compare its wcs later
	m  = m0
	# Submap slicing currently has wrapping issues
	if args.sub is not None:
		default = [[-90,-180],[90,180]]
		sub  = np.array([[(default[j][i] if q == '' else float(q))*np.pi/180 for j,q in enumerate(w.split(":"))]for i,w in enumerate(args.sub.split(","))]).T
		m = m.submap(sub)
	# Perform a common autocrop across all fields
	if args.autocrop:
		m = enmap.autocrop(m)
	# Downgrade
	downgrade = [int(w) for w in args.downgrade.split(",")]
	m = enmap.downgrade(m, downgrade)
	# Slicing, either at the file name level or though the slice option
	m = eval("m"+slice)
	if args.slice is not None:
		m = eval("m"+args.slice)
	flip = (m.wcs.wcs.cdelt*m0.wcs.wcs.cdelt)[::-1]<0
	assert m.ndim >= 2, "Image must have at least 2 dimensions"
	# Apply arbitrary map operations
	if args.op is not None:
		m = eval(args.op, {"m":m},np.__dict__)
	# Scale if requested
	scale = [int(w) for w in args.scale.split(",")]
	if np.any(np.array(scale)>1):
		m = enmap.upgrade(m, scale)
	# Flatten pre-dimensions
	mf = m.reshape((-1,)+m.shape[-2:])
	# Mask bad data
	if args.mask is not None:
		if not np.isfinite(args.mask): mf[mf==args.mask] = np.nan
		else: mf[np.abs(mf-args.mask)<=args.mask_tol] = np.nan
	# Flip such that pixels are in PIL or matplotlib convention,
	# such that RA increases towards the left and dec upwards in
	# the final image. Unless a slicing operation on the image
	# overrrode this.
	if mf.wcs.wcs.cdelt[1] > 0: mf = mf[:,::-1,:]
	if mf.wcs.wcs.cdelt[0] > 0: mf = mf[:,:,::-1]
	if flip[0]: mf = mf[:,::-1,:]
	if flip[1]: mf = mf[:,:,::-1]
	# Done
	return mf

def draw_map_field(map, args, crange=None, return_layers=False, return_info=False, printer=noprint):
	"""Draw a single map field, resulting in a single image. Adds a coordinate grid
	and lables as specified by args. If return_layers is True, an array will be
	returned instead of an image, wich each entry being a component of the image,
	such as the base image, the coordinate grid, the labels, etc. If return_bounds
	is True, then the """
	map = prepare_map_field(map, args, crange, printer=printer)
	layers = []
	names  = []
	# Image layer
	with printer.time("to image", 3):
		img = PIL.Image.fromarray(utils.moveaxis(map,0,2)).convert('RGBA')
	layers.append((img,[[0,0],img.size]))
	names.append("img")
	# Coordinate grid
	if args.grid % 2:
		with printer.time("draw grid", 3):
			ginfo = calc_gridinfo(map.shape, map.wcs, args)
			layers.append(draw_grid(ginfo, args))
			names.append("grid")
		if not args.nolabels:
			with printer.time("draw labels", 3):
				layers.append(draw_grid_labels(ginfo, args))
				names.append("tics")
	# Possibly other stuff too, like point source circles
	# or contours
	with printer.time("stack layers", 3):
		layers, bounds = standardize_images(layers)
		if not return_layers: layers = merge_images(layers)
	class Info:
		def __init__(self, bounds, names):
			self.bounds = bounds
			self.names  = names
	info = Info(bounds, names)
	if return_info: return layers, info
	else: return layers

def draw_map_field_mpl(map, args, crange=None, printer=noprint):
	"""Render a map field using matplotlib. Less tested and
	maintained than draw_map_field, and supports fewer features.
	Returns an object one can call savefig on to draw."""
	map = prepare_map_field(map, args, crange, printer=printer)
	# Set up matplotlib. We do it locally here to
	# avoid having it as a dependency in general
	with printer.time("matplotplib", 3):
		import matplotlib
		matplotlib.use("Agg")
		from matplotlib import pyplot, ticker
		matplotlib.rcParams.update({'font.size': 10})
		dpi, pad = args.mpl_dpi, args.mpl_pad
		winch, hinch = map.shape[2]/dpi, map.shape[1]/dpi
		fig  = pyplot.figure(figsize=(winch+pad,hinch+pad))
		box  = map.box()*180/np.pi
		pyplot.imshow(utils.moveaxis(map,0,2), extent=[box[0,1],box[1,1],box[1,0],box[0,0]])
		# Make conformal in center of image
		pyplot.axes().set_aspect(1/np.cos(np.mean(map.box()[:,0])))
		if args.grid % 2:
			ax = pyplot.axes()
			ticks = np.full(2,1.0); ticks[:] = [float(w) for w in args.ticks.split(",")]
			ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[1]))
			ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[0]))
			if args.subticks:
				ax.xaxis.set_minor_locator(ticker.MultipleLocator(args.sub))
				ax.yaxis.set_minor_locator(ticker.MultipleLocator(args.sub))
				pyplot.minorticks_on()
				pyplot.grid(True, which="major", linewidth=2)
				pyplot.grid(True, which="minor", linewidth=1)
			else:
				pyplot.grid(True)
		pyplot.tight_layout(pad=0.0,h_pad=0.0,w_pad=0.0)
	return pyplot
	#pyplot.savefig(oname,bbox_inches="tight",dpi=dpi)

def parse_range(desc,n):
	res = np.array([float(w) for w in desc.split(":")])[:n]
	return np.concatenate([res,np.repeat([res[-1]],n-len(res))])

def get_color_range(map, args):
	"""Compute an appropriate color bare range from map[:,ny,nx]
	given the args. Relevant members are range, min, max, quantile."""
	# Construct color ranges
	ncomp  = map.shape[0]
	crange = np.zeros((2,ncomp))+np.nan
	# Try explicit limits if given
	if args.range is not None:
		crange[1] = parse_range(args.range,ncomp)
		crange[0] = -crange[1]
	if args.min is not None: crange[0] = parse_range(args.min,ncomp)
	if args.max is not None: crange[1] = parse_range(args.max,ncomp)
	# Fall back on quantile otherwise
	if np.any(np.isnan(crange)):
		vals = np.sort(map[np.isfinite(map)])
		n    = len(vals)
		v1   = vals[int(round(n*args.quantile))]
		v2   = vals[min(n-1,int(round(n*(1-args.quantile))))]
		crange[0,np.isnan(crange[0])] = v1
		crange[1,np.isnan(crange[1])] = v2
	return crange

def get_num_digits(n): return int(np.log10(n))+1
def split_file_name(fname):
	"""Split a file name into directory, base name and extension,
	such that fname = dirname + "/" + basename + "." + ext."""
	dirname  = os.path.dirname(fname)
	if len(dirname) == 0: dirname = "."
	base_ext = os.path.basename(fname)
	# Find the extension. Using the last dot does not work for .fits.gz.
	# Using the first dot in basename does not work for foo2.5_bar.fits.
	# Will therefore handle .gz as a special case.
	if base_ext.endswith(".gz"):
		dot = base_ext[:-3].rfind(".")
	else:
		dot  = base_ext.rfind(".")
	if dot < 0: dot = len(base_ext)
	base = base_ext[:dot]
	ext  = base_ext[dot+1:]
	return dirname, base, ext

def map_to_color(map, crange, args):
	"""Compute an [{R,G,B},ny,nx] color map based on a map[1 or 3, ny,nx]
	map and a corresponding color range crange[{min,max}]. Relevant args
	fields: color, method, rgb. If rgb is not true, only the first element
	of the input map will be used. Otherwise 3 will be used."""
	map = (map-crange[0])/(crange[1]-crange[0])
	if args.rgb: m_color = colorize.colorize(map,    desc=args.color, method="direct")
	else:        m_color = colorize.colorize(map[0], desc=args.color, method=args.method)
	m_color = enmap.samewcs(np.rollaxis(m_color,2), map)
	return m_color

def calc_gridinfo(shape, wcs, args):
	"""Compute the points making up the grid lines for the given map.
	Depends on args.ticks and args.nstep."""
	ticks = np.full(2,1.0); ticks[:] = [float(w) for w in args.ticks.split(",")]
	return cgrid.calc_gridinfo(shape, wcs, steps=ticks, nstep=args.nstep)

def draw_grid(ginfo, args):
	"""Return a grid based on gridinfo. args.grid_color controls the color
	the grid will be drawn with."""
	grid = cgrid.draw_grid(ginfo, color=args.grid_color)
	bounds = np.array([[0,0],ginfo.shape[::-1]])
	return grid, bounds

def draw_grid_labels(ginfo, args):
	"""Return an image with a coordinate grid, along with abounds of this
	image relative to the coordinate shape stored in ginfo. Depends
	on the following args members: args.font, args.font_size, args.font_color"""
	linfo = []
	for gi in [ginfo.lat, ginfo.lon]:
		linfo += cgrid.calc_label_pos(gi, ginfo.shape[::-1])
	canvas = PIL.Image.new("RGBA", ginfo.shape[::-1])
	labels, bounds = cgrid.draw_labels(canvas, linfo, fname=args.font, fsize=args.font_size, color=args.font_color, return_bounds=True)
	return labels, bounds

def standardize_images(tuples):
	"""Given a list of (img,bounds), composite them on top of each other
	(first at the bottom), and return the total image and its new bounds."""
	bounds_all = np.array([bounds for img, bounds in tuples])
	bounds_full= cgrid.calc_bounds(bounds_all, tuples[0][1][1])
	# Build canvas
	totsize = bounds_full[1]-bounds_full[0]
	res = []
	for img, bounds in tuples:
		# Expand to full size
		img_big = PIL.Image.new("RGBA", totsize)
		img_big.paste(img, tuple(bounds[0]-bounds_full[0]))
		res.append(img_big)
	return res, bounds_full

def merge_images(images):
	"""Stack all images into an alpha composite. Images must all have consistent
	extent before this. Use standardize_images to achieve this."""
	res = images[0]
	for img in images[1:]:
		res = PIL.Image.alpha_composite(res, img)
	return res

def prepare_map_field(map, args, crange=None, printer=noprint):
	if crange is None:
		with printer.time("ranges", 3):
			crange = get_color_range(map, args)
	if map.ndim == 2:
		map = map[None]
	if args.autocrop_each:
		map = enmap.autocrop(map)
	with printer.time("colorize", 3):
		map = map_to_color(map, crange, args)
	return map

class dprint:
	def __init__(self, desc, args):
		self.desc = desc
		self.args = args
	def __enter__(self):
		self.t1 = time.time()
	def __exit__(self, type, value, traceback):
		if self.args.verbosity >= 3:
			sys.stderr.write("%6.2f %s\n" % (time.time()-self.t1,self.desc))
