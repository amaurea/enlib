import numpy as np, argparse, time, sys, warnings, os, shlex, glob, PIL.Image, PIL.ImageDraw
from scipy import ndimage
from enlib import enmap, colorize, mpi, cgrid, utils, memory, bunch, wcs as enwcs
# Optional dependency array_ops needed for contour drawing
try: from enlib import array_ops
except ImportError: pass

class Printer:
	def __init__(self, level=1, prefix=""):
		self.level  = level
		self.prefix = prefix
	def write(self, desc, level, exact=False, newline=True, prepend=""):
		if level == self.level or not exact and level <= self.level:
			prepend = "%6.2f " % (memory.max()/1024.**3) + prepend
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
	cache = {}
	# Plot each file
	for fi in range(comm.rank,len(ifiles),comm.size):
		ifile = ifiles[fi]
		with printer.time("read %s" % ifile, 3):
			map, minfo = get_map(ifile, args, return_info=True)
		with printer.time("ranges", 3):
			crange= get_color_range(map, args)
		for ci, cr in enumerate(crange.T):
			printer.write("color range %d: %12.5e to %15.7e" % (ci, cr[0], cr[1]), 4)
		# Loop over map fields
		ncomp  = map.shape[0]
		ngroup = 3 if args.rgb else 1
		crange_ind = 0
		for i in range(0, ncomp, ngroup):
			# The unflattened index of the current field
			N = minfo.ishape[:-2]
			I = np.unravel_index(i, N) if len(N) > 0 else []
			if args.symmetric and np.any(np.sort(I) != I):
				continue
			# Construct default out format
			ndigit   = get_num_digits(ncomp)
			ndigits  = [get_num_digits(n) for n in N]
			subprint = printer.push(("%%0%dd/%%d " % ndigit) % (i+1,ncomp))
			dir, base, ext = split_file_name(minfo.fname)
			map_field = map[i:i+ngroup]
			if minfo.wcslist:
				# HACK: If stamp extraction messed with the wcs, fix it here
				map_field.wcs = minfo.wcslist[I[0]]
			# Build output file name
			oinfo = {"dir":"" if dir == "." else dir + "/", "base":base, "iext":ext,
					"fi":fi, "fn":len(args.ifiles), "ci":i, "cn":ncomp, "pi":comm.rank, "pn":comm.size,
					"pre":args.prefix, "suf":args.suffix,
					"comp": "_"+"_".join(["%0*d" % (ndig,ind) for ndig,ind in zip(ndigits,I)]) if len(N) > 0 else "",
					"fcomp": "_%0*d" % (ndigit,i) if len(minfo.ishape) > 2 else "",
					"ext":args.ext, "layer":""}
			oname = args.oname.format(**oinfo)
			# Draw the map
			if args.driver.lower() == "pil":
				img, info = draw_map_field(map_field, args, crange[:,crange_ind:crange_ind+ngroup], return_info=True, return_layers=args.layers, printer=subprint, cache=cache)
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
				figure = draw_map_field_mpl(map_field, args, crange[:,crange_ind:crange_ind+ngroup], printer=subprint)
				with subprint.time("write to %s" % oname, 3):
					figure.savefig(oname,bbox_inches="tight",dpi=args.mpl_dpi)
			# Progress report
			printer.write("\r%s %5d/%d" % (ifile, i+1,ncomp), 2, exact=True, newline=False)
			crange_ind += 1
		printer.write("",    2, exact=True)
		printer.write(ifile, 1, exact=True)

def parse_args(args=sys.argv[1:], noglob=False):
	parser = argparse.ArgumentParser()
	parser.add_argument("ifiles", nargs="+", help="The map files to plot. Each file will be processed independently and output as an image file with a name derived from that of the input file (see --oname). For each file a color range will be determined, and each component of the map (if multidimensional) will be written to a separate image file. If the file has more than 1 non-pixel dimension, these will be flattened first.")
	parser.add_argument("-o", "--oname", default="{dir}{pre}{base}{suf}{comp}{layer}.{ext}", help="The format to use for the output name. Default is {dir}{pre}{base}{suf}{comp}{layer}.{ext}")
	parser.add_argument("-c", "--color", default="planck", help="The color scheme to use, e.g. planck, wmap, gray, hotcold, etc., or a colors pecification in the form val:rrggbb,val:rrggbb,.... Se enlib.colorize for details.")
	parser.add_argument("-r", "--range", type=str, help="The symmetric color bar range to use. If specified, colors in the map will be truncated to [-range,range]. To give each component in a multidimensional map different color ranges, use a colon-separated list, for example -r 250:100:50 would plot the first component with a range of 250, the second with a range of 100 and the third and any subsequent component with a range of 50.")
	parser.add_argument("--min", type=str, help="The value at which the color bar starts. See --range.")
	parser.add_argument("--max", type=str, help="The value at which the color bar ends. See --range.")
	parser.add_argument("-q", "--quantile", type=float, default=0.01, help="Which quantile to use when automatically determining the color range. If specified, the color bar will go from [quant(q),quant(1-q)].")
	parser.add_argument("-v", dest="verbosity", action="count", help="Verbose output. Specify multiple times to increase verbosity further.")
	parser.add_argument("-s", "-u", "--scale", "--upgrade", type=str, default="1", help="Upscale the image using nearest neighbor interpolation by this amount before plotting. For example, 2 would make the map twice as large in each direction, while 4,1 would make it 4 times as tall and leave the width unchanged.")
	parser.add_argument("--verbosity", dest="verbosity", type=int, help="Specify verbosity directly as an integer.")
	parser.add_argument("--method", default="auto", help="Which colorization implementation to use: auto, fortran or python.")
	parser.add_argument("--slice", type=str, help="Apply this numpy slice to the map before plotting.")
	parser.add_argument("--sub",   type=str, help="Slice a map based on dec1:dec2,ra1:ra2.")
	parser.add_argument("--op", type=str, help="Apply this general operation to the map before plotting. For example, 'log(abs(m))' would give you a lograithmic plot.")
	parser.add_argument("-d", "--downgrade", type=str, default="1", help="Downsacale the map by this factor before plotting. This is done by averaging nearby pixels. See --scale for syntax.")
	parser.add_argument("--prefix", type=str, default="", help="Specify a prefix for the output file. See --oname.")
	parser.add_argument("--suffix", type=str, default="", help="Specify a suffix for the output file. See --oname.")
	parser.add_argument("--ext", type=str, default="png", help="Specify an extension for the output file. This will determine the file type of the resulting image. Can be anything PIL recognizes. The default is png.")
	parser.add_argument("-m", "--mask", type=float, help="Mask this value, making it transparent in the output image. For example -m 0 would mark all values exactly equal to zero as missing.")
	parser.add_argument("--mask-tol", type=float, default=1e-14, help="The tolerance to use with --mask.")
	parser.add_argument("-g", "--grid", action="count", default=1, help="Toggle the coordinate grid. Disabling it can make plotting much faster when plotting many small maps.")
	parser.add_argument("--grid-color", type=str, default="00000020", help="The RGBA color to use for the grid.")
	parser.add_argument("-t", "--ticks", type=str, default="1", help="The grid spacing in degrees. Either a single number to be used for both axis, or ty,tx.")
	parser.add_argument("--nolabels", action="store_true", help="Disable the generation of coordinate labels outside the map when using the grid.")
	parser.add_argument("--nstep", type=int, default=200, help="The number of steps to use when drawing grid lines. Higher numbers result in smoother curves.")
	parser.add_argument("--subticks", type=float, default=0, help="Subtick spacing. Only supported by matplotlib driver.")
	parser.add_argument("--font", type=str, default="arial.ttf", help="The font to use for text.")
	parser.add_argument("--font-size", type=int, default=20, help="Font size to use for text.")
	parser.add_argument("--font-color", type=str, default="000000", help="Font color to use for text.")
	parser.add_argument("-D", "--driver", type=str, default="pil", help="The driver to use for plotting. Can be pil (the default) or mpl. pil cleanly maps input pixels to output pixels, and has better coordiante system support, but doesn't have as pretty grid lines or axis labels.")
	parser.add_argument("--mpl-dpi", type=float, default=75, help="The resolution to use for the mpl driver.")
	parser.add_argument("--mpl-pad", type=float, default=1.6, help="The padding to use for the mpl driver.")
	parser.add_argument("--rgb", action="store_true", help="Enable RGB mode. The input maps must have 3 components, which will be interpreted as red, green and blue channels of a single image instead of 3 separate images as would be the case without this option. The color scheme is overriden in this case.")
	parser.add_argument("--reverse-color",  action="store_true", help="Reverse the color scale. For example, a black-to-white scale will become a white-to-black sacle.")
	parser.add_argument("-a", "--autocrop", action="store_true", help="Automatically crop the image by removing expanses of uniform color around the edges. This is done jointly for all components in a map, making them directly comparable, but is done independently for each input file.")
	parser.add_argument("-A", "--autocrop-each", action="store_true", help="As --autocrop, but done individually for each component in each map.")
	parser.add_argument("-L", "--layers", action="store_true", help="Output the individual layers that make up the final plot (such as the map itself, the coordinate grid, the axis labels, any contours and lables) as individual files instead of compositing them into a final image.")
	parser.add_argument("-C", "--contours", type=str, default=None, help="Enable contour lines. For example -C 10 to place a contour at every 10 units in the map, -C 5:10 to place it at every 10 units, but starting at 5, and 1,2,4,8 or similar to place contours at manually chosen locations.")
	parser.add_argument("--contour-color", type=str, default="000000", help="The color scheme to use for contour lines. Either a single rrggbb, a val:rrggbb,val:rrggbb,... specification or a color scheme name, such as planck, wmap or gray.")
	parser.add_argument("--contour-width", type=int, default=1, help="The width of each contour line, in pixels.")
	parser.add_argument("--annotate",      type=str, default=None, help="""Annotate the map with text, lines or circles. Should be a text file with one entry per line, where an entry can be:
		c[ircle] lat lon dy dx [rad [width [color]]]
		t[ext]   lat lon dy dx text [size [color]]
		l[ine]   lat lon dy dx lat lon dy dx [width [color]]
	dy and dx are pixel-unit offsets from the specified lat/lon.""")
	parser.add_argument("--stamps", type=str, default=None, help="Plot stamps instead of the whole map. Format is srcfile:size:nmax, where the last two are optional. srcfile is a file with [dec ra] in degrees, size is the size in pixels of each stamp, and nmax is the max number of stamps to produce.")
	parser.add_argument("--tile",  type=str, default=None, help="Stack components vertically and horizontally. --tile 5,4 stacks into 5 rows and 4 columns. --tile 5 or --tile 5,-1 stacks into 5 rows and however many columns are needed. --tile -1,5 stacks into 5 columns and as many rows are needed. --tile -1 allocates both rows and columns to make the result as square as possible. The result is treated as a single enmap, so the wcs will only be right for one of the tiles.")
	parser.add_argument("--tile-transpose", action="store_true", help="Transpose the ordering of the fields when tacking. Normally row-major stacking is used. This sets column-major order instead.")
	parser.add_argument("-S", "--symmetric", action="store_true", help="Treat the non-pixel axes as being asymmetric matrix, and only plot a non-redundant triangle of this matrix.")
	parser.add_argument("-z", "--zenith",    action="store_true", help="Plot the zenith angle instead of the declination.")
	parser.add_argument("-F", "--fix-wcs",   action="store_true", help="Fix the wcs for maps in cylindrical projections where the reference point was placed too far away from the map center.")
	if isinstance(args, basestring):
		args = shlex.split(args)
	res = parser.parse_args(args)
	res = bunch.Bunch(**res.__dict__)
	# Glob expansion
	if not noglob:
		ifiles = []
		for pattern in res.ifiles:
			matches = glob.glob(pattern)
			if len(matches) > 0:
				ifiles += matches
			else:
				ifiles.append(pattern)
		res.ifiles = ifiles
	return res

def get_map(ifile, args, return_info=False):
	"""Read the specified map, and massage it according to the options
	in args. Relevant ones are sub, autocrop, slice, op, downgrade, scale,
	mask. Retuns with shape [:,ny,nx], where any extra dimensions have been
	flattened into a single one."""
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		toks = ifile.split(":")
		ifile, slice = toks[0], ":".join(toks[1:])
		m0 = enmap.read_map(ifile)
		if args.fix_wcs:
			m0.wcs = enwcs.fix_wcs(m0.wcs)
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
		# If necessary, split into stamps. If no stamp splitting occurs,
		# a list containing only the original map is returned
		mlist = extract_stamps(m, args)
		# The stamp stuff is a bit of an ugly hack. This loop and wcslist
		# are parts of that hack.
		for i, m in enumerate(mlist):
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
			m1 = m
			if args.op is not None:
				m = eval(args.op, {"m":m},np.__dict__)
			# Scale if requested
			scale = [int(w) for w in args.scale.split(",")]
			if np.any(np.array(scale)>1):
				m = enmap.upgrade(m, scale)
			# Flip such that pixels are in PIL or matplotlib convention,
			# such that RA increases towards the left and dec upwards in
			# the final image. Unless a slicing operation on the image
			# overrrode this.
			if m.wcs.wcs.cdelt[1] > 0: m = m[...,::-1,:]
			if m.wcs.wcs.cdelt[0] > 0: m = m[...,:,::-1]
			if flip[0]: m = m[...,::-1,:]
			if flip[1]: m = m[...,:,::-1]
			# Update stamp list
			mlist[i] = m
		wcslist = [m.wcs for m in mlist]
		m = enmap.samewcs(np.asarray(mlist),mlist[0])
		if args.stamps is None:
			m, wcslist = m[0], None
		# Flatten pre-dimensions
		mf = m.reshape((-1,)+m.shape[-2:])
		# Stack
		if args.tile is not None:
			toks = [int(i) for i in args.tile.split(",")]
			nrow = toks[0] if len(toks) > 0 else -1
			ncol = toks[1] if len(toks) > 1 else -1
			mf = hwstack(hwexpand(mf, nrow, ncol, args.tile_transpose))[None]
		# Mask bad data
		if args.mask is not None:
			if not np.isfinite(args.mask): mf[np.abs(mf)==args.mask] = np.nan
			else: mf[np.abs(mf-args.mask)<=args.mask_tol] = np.nan
		# Done
		if not return_info: return mf
		else:
			info = bunch.Bunch(fname=ifile, ishape=m.shape, wcslist=wcslist)
			return mf, info

def extract_stamps(map, args):
	"""Given a map, extract a set of identically sized postage stamps based on
	args.stamps. Returns a new map consisting of a stack of these stamps, along
	with a list of each of these' wcs object."""
	if args.stamps is None: return [map]
	# Stamps specified by format srcfile[:size[:nmax]], where the srcfile has
	# lines of [dec, ra] in degrees
	toks = args.stamps.split(":")
	# Read in our positions, optionally truncating the list
	srcs = np.loadtxt(toks[0]).T[:2]*utils.degree
	size = int(toks[1]) if len(toks) > 1 else 16
	nsrc = int(toks[2]) if len(toks) > 2 else len(srcs.T)
	srcs = srcs[:,:nsrc]
	# Convert to pixel coordinates of corners
	pix  = np.round(map.sky2pix(srcs)-0.5*size).astype(int)
	# Extract stamps
	return map.stamps(pix.T, size, aslist=True)

def get_cache(cache, key, fun):
	if cache is None: return fun()
	if key not in cache: cache[key] = fun()
	return cache[key]

def draw_map_field(map, args, crange=None, return_layers=False, return_info=False, printer=noprint, cache=None):
	"""Draw a single map field, resulting in a single image. Adds a coordinate grid
	and lables as specified by args. If return_layers is True, an array will be
	returned instead of an image, wich each entry being a component of the image,
	such as the base image, the coordinate grid, the labels, etc. If return_bounds
	is True, then the """
	map, color = prepare_map_field(map, args, crange, printer=printer)
	tag    = (tuple(map.shape), map.wcs.to_header_string(), repr(args))
	layers = []
	names  = []
	# Image layer
	with printer.time("to image", 3):
		img = PIL.Image.fromarray(utils.moveaxis(color,0,2)).convert('RGBA')
	layers.append((img,[[0,0],img.size]))
	names.append("img")
	# Contours
	if args.contours:
		with printer.time("draw contours", 3):
			contour_levels = calc_contours(crange, args)
			cimg = draw_contours(map, contour_levels, args)
			layers.append((cimg, [[0,0],cimg.size]))
			names.append("cont")
	# Annotations
	if args.annotate:
		with printer.time("draw annotations", 3):
			def get_aimg():
				annots = parse_annotations(args.annotate)
				return draw_annotations(map, annots, args)
			aimg = get_cache(cache, ("annotate",tag), get_aimg)
			layers.append((aimg, [[0,0],aimg.size]))
			names.append("annot")
	# Coordinate grid
	if args.grid % 2:
		with printer.time("draw grid", 3):
			ginfo = get_cache(cache, ("ginfo",tag), lambda: calc_gridinfo(map.shape, map.wcs, args))
			grid  = get_cache(cache, ("grid", tag), lambda: draw_grid(ginfo, args))
			layers.append(grid)
			names.append("grid")
		if not args.nolabels:
			with printer.time("draw labels", 3):
				labels = get_cache(cache, ("labels",tag), lambda: draw_grid_labels(ginfo, args))
				layers.append(labels)
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
	map, color = prepare_map_field(map, args, crange, printer=printer)
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
		pyplot.imshow(utils.moveaxis(color,0,2), extent=[box[0,1],box[1,1],box[1,0],box[0,0]])
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
	map = ((map.T-crange[0])/(crange[1]-crange[0])).T # .T ensures broadcasting for rgb case
	if args.reverse_color: map = 1-map
	if args.rgb: m_color = colorize.colorize(map,    desc=args.color, driver=args.method, mode="direct")
	else:        m_color = colorize.colorize(map[0], desc=args.color, driver=args.method)
	m_color = enmap.samewcs(np.rollaxis(m_color,2), map)
	return m_color

def calc_gridinfo(shape, wcs, args):
	"""Compute the points making up the grid lines for the given map.
	Depends on args.ticks and args.nstep."""
	ticks = np.full(2,1.0); ticks[:] = [float(w) for w in args.ticks.split(",")]
	return cgrid.calc_gridinfo(shape, wcs, steps=ticks, nstep=args.nstep, zenith=args.zenith)

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

def calc_contours(crange, args):
	"""Returns a list of values at which to place contours based on
	the valure range of the map crange[{from,to}] and the contour
	specification in args.

	Contour specifications:
		base:step or val,val,val...
	base: number
	step: number (explicit), -number (relative)
	"""
	if args.contours is None: return None
	vals = args.contours.split(",")
	if len(vals) > 1:
		# Explicit values
		return np.array([float(v) for v in vals if len(v) > 0])
	else:
		# base:step
		toks = args.contours.split(":")
		if len(toks) == 1:
			base, step = 0, float(toks[0])
		else:
			base, step = float(toks[0]), float(toks[1])
		if step < 0:
			step = (crange[1]-crange[0])/(-step)
		# expand to fill crange
		a = int(np.ceil ((crange[0]-base)/step))
		b = int(np.floor((crange[1]-base)/step))+1
		return np.arange(a,b)*step + base

def draw_contours(map, contours, args):
	img   = PIL.Image.new("RGBA", map.shape[-2:][::-1])
	inds  = np.argsort(contours)
	cmap  = array_ops.find_contours(map[0], contours[inds])
	cmap  = contour_widen(cmap, args.contour_width)
	cmap -= 1
	# Undo sorting if we sorted
	if not np.allclose(inds, np.arange(len(inds))):
		mask = cmap>=0
		cmap[mask] = inds[cmap[mask]]
	cmap  = cmap.astype(float)
	# Make non-contour areas transparent
	cmap[cmap<0] = np.nan
	# Rescale to 0:1
	if len(contours) > 1:
		cmap /= len(contours)-1
	color = colorize.colorize(cmap, desc=args.contour_color, driver=args.method)
	return PIL.Image.fromarray(color).convert('RGBA')

def parse_annotations(afile):
	with open(afile,"r") as f:
		return [shlex.split(line) for line in f]

def draw_annotations(map, annots, args):
	"""Draw a set of annotations on the map. These are specified
	as a list of ["type",param,param,...]. The recognized formats
	are:
		c[ircle] lat lon dy dx [rad [width [color]]]
		t[ext]   lat lon dy dx text [size [color]]
		l[ine]   lat lon dy dx lat lon dy dx [width [color]]
		r[ect]   lat lon dy dx lat lon dy dx [width [color]]
	dy and dx are pixel-unit offsets from the specified lat/lon.
	This is useful for e.g. placing text next to circles."""
	img  = PIL.Image.new("RGBA", map.shape[-2:][::-1])
	draw = PIL.ImageDraw.Draw(img, "RGBA")
	font = None
	font_size_prev = 0
	def topix(pos_off):
		pix = map.sky2pix(np.array([float(w) for w in pos_off[:2]])*utils.degree)
		pix += np.array([float(w) for w in pos_off[2:]])
		return pix[::-1].astype(int)
	for annot in annots:
		atype = annot[0].lower()
		color = "black"
		width = 2
		if atype in ["c","circle"]:
			x,y = topix(annot[1:5])
			rad = 8
			if len(annot) > 5: rad   = int(annot[5])
			if len(annot) > 6: width = int(annot[6])
			if len(annot) > 7: color = annot[7]
			antialias = 1 if width < 1 else 4
			draw_ellipse(img,
					(x-rad,y-rad,x+rad,y+rad),
					outline=color,width=width, antialias=antialias)
		elif atype in ["l","line"] or atype in ["r","rect"]:
			x1,y1 = topix(annot[1:5])
			x2,y2 = topix(annot[5:9])
			if x2 < x1: x1,x2 = x2,x1
			if y2 < y1: y1,y2 = y2,y1
			print x1, y1, x2, y2
			if len(annot) >  9: width = int(annot[9])
			if len(annot) > 10: color = annot[10]
			if atype[0] == "l":
				draw.line((x1,y1,x2,y2), fill=color, width=width)
			else:
				for i in range(width):
					draw.rectangle((x1+i,y1+i,x2-i,y2-i), outline=color)
		elif atype in ["t", "text"]:
			x,y  = topix(annot[1:5])
			text = annot[5]
			size = 16
			if len(annot) > 6: size  = int(annot[6])
			if len(annot) > 7: color = annot[7]
			if font is None or size != font_size_prev:
				font = cgrid.get_font(size)
				font_size_prev = size
			tbox = font.getsize(text)
			draw.text((x-tbox[0]/2, y-tbox[1]/2), text, color, font=font)
		else:
			raise NotImplementedError
	return img

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
		img_big = PIL.Image.new("RGBA", tuple(totsize))
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
		color = map_to_color(map, crange, args)
	return map, color

def makefoot(n):
	b = np.full((2*n+1,2*n+1),1)
	b[n,n] = 0
	b = ndimage.distance_transform_edt(b)
	return b[1::2,1::2] < n

def contour_widen(cmap, width):
	if width <= 1: return cmap
	foot = makefoot(width)
	return ndimage.grey_dilation(cmap, footprint=foot)

def draw_ellipse(image, bounds, width=1, outline='white', antialias=1):
	"""Improved ellipse drawing function, based on PIL.ImageDraw.
	Improved from
	http://stackoverflow.com/questions/32504246/draw-ellipse-in-python-pil-with-line-thickness"""
	bounds = np.asarray(bounds)
	# Create small coordinate system around ellipse, with a
	# margin of width on each side
	esize  = bounds[2:]-bounds[:2] + 2*width
	ebounds= bounds - bounds[[0,1,0,1]] + width
	# Use a single channel image (mode='L') as mask.
	# The size of the mask can be increased relative to the imput image
	# to get smoother looking results. 
	mask = PIL.Image.new(size=tuple(esize*antialias), mode='L', color='black')
	draw = PIL.ImageDraw.Draw(mask)
	# draw outer shape in white (color) and inner shape in black (transparent)
	for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
		a = (ebounds[:2] + offset)*antialias
		b = (ebounds[2:] - offset)*antialias
		draw.ellipse([a[0],a[1],b[0],b[1]], fill=fill)
	# downsample the mask using PIL.Image.LANCZOS 
	# (a high-quality downsampling filter).
	mask = mask.resize(esize, PIL.Image.LANCZOS)
	# paste outline color to input image through the mask
	image.paste(outline, tuple(bounds[:2]-width), mask=mask)

def hwexpand(mflat, nrow=-1, ncol=-1, transpose=False):
	"""Stack the maps in mflat[n,ny,nx] into a single flat map mflat[nrow,ncol,ny,nx]"""
	n, ny, nx = mflat.shape
	if nrow < 0 and ncol < 0:
		ncol = int(np.ceil(n**0.5))
	if nrow < 0: nrow = (n+ncol-1)/ncol
	if ncol < 0: ncol = (n+nrow-1)/nrow
	if not transpose:
		omap = enmap.zeros([nrow,ncol,ny,nx],mflat.wcs,mflat.dtype)
		omap.reshape(-1,ny,nx)[:n] = mflat
	else:
		omap = enmap.zeros([ncol,nrow,ny,nx],mflat.wcs,mflat.dtype)
		omap.reshape(-1,ny,nx)[:n] = mflat
		omap = np.transpose(omap,(1,0,2,3))
	return omap

def hwstack(mexp):
	nr,nc,ny,nx = mexp.shape
	return np.transpose(mexp,(0,2,1,3)).reshape(nr*ny,nc*nx)
