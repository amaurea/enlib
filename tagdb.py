"""This module provides a class that lets one associate tags and values
with a set of ids, and then select ids based on those tags and values.
For example query("deep56,night,ar2,in(bounds,Moon)") woult return a
set of ids with the tags deep56, night and ar2, and where th Moon[2]
array is in the polygon specified by the bounds [:,2] array."""
import re, numpy as np, h5py, shlex, copy, warnings, time, os
from . import utils

class Tagdb:
	def __init__(self, data=None, sort="id", default_fields=[], default_query=""):
		"""Most basic constructor. Takes a dictionary
		data[name][...,nid], which must contain the field "id"."""
		if data is None:
			self.data = {"id":np.zeros(0,dtype='S5')}
		else:
			self.data = {key:np.array(val) for key,val in data.iteritems()}
			assert "id" in self.data, "Id field missing"
			if self.data["id"].size == 0: self.data["id"] = np.zeros(0,dtype='S5')
		# Inser default fields. These will always be present, but will be
		for field_expr in default_fields:
			if isinstance(field_expr, basestring): field_expr = (field_expr,)
			name  = field_expr[0]
			value = field_expr[1] if len(field_expr) > 1 else False
			dtype = field_expr[2] if len(field_expr) > 2 else type(value)
			if name not in self.data:
				self.data[name] = np.full(len(self), value, dtype=dtype)
		# Register the default query. This will be suffixed to each query.
		self.default_query = default_query
		# Set our sorting field
		self.sort  = sort
		self.functors = {}
		self.add_func("file_contains", file_contains)
	def add_func(self, name, func):
		self.functors[name] = lambda data: func
	def add_functor(self, name, init):
		self.functors[name] = init
	def copy(self):
		return copy.deepcopy(self)
	@property
	def ids(self): return self.data["id"]
	@property
	def tags(self):
		return sorted(self.data.keys())
	def __len__(self): return len(self.ids)
	def __getitem__(self, query=""):
		return self.query(query)
	def select(self, ids):
		"""Return a tagdb which only contains the selected ids."""
		if isinstance(ids, basestring):
			ids = self.query(ids)
		ids = np.asarray(ids)
		if issubclass(ids.dtype.type, np.integer):
			# Fast integer slicing
			return self.__class__(dslice(self.data, ids))
		else:
			# Slice by id
			# Restrict to the subset of these ids
			inds = utils.find(self.ids, ids)
			odata = dslice(self.data, inds)
			res = self.copy()
			res.data = odata
			return res
	def query(self, query=None, apply_default_query=True):
		"""Query the database. The query takes the form
		tag,tag,tag,...:sort[slice], where all tags must be satisfied for an id to
		be returned. More general syntax is also available. For example,
		(a+b>c)|foo&bar,cow. This follows standard python and numpy syntax,
		except that , is treated as a lower-priority version of &."""
		# Make a copy of self.data so we can't modify it without changing ourself
		data = self.data.copy()
		# First split off any sorting field or slice
		if query is None: query = ""
		toks = utils.split_outside(query,":")
		query, rest = toks[0], ":".join(toks[1:])
		# Hack: Support id fields as tags, even if they contain
		# illegal characters..
		t1 = time.time()
		for id in data["id"]:
			if id not in query: continue
			query = re.sub(r"""(?<!['"])\b%s\b""" % id, "(id=='%s')" % id, query)
		# Split into ,-separated fields.
		toks = utils.split_outside(query,",")
		fields = []
		override_ids = None
		for tok in toks:
			# We don't support subid tags any more. These were used to handle both
			# frequency selection and arbitrary array subsets. We now handle frequency
			# selection via normal tags, so translate subid tags to normal tags to
			# let old selectors keep working. Arbitrary array subsets are no longer
			# supported, sadly.
			tok = tok.lstrip("+")
			if len(tok) == 0: continue
			if tok.startswith("/"):
				# Tags starting with / will be interpreted as special query flags
				if tok == "/all": apply_default_query = False
				else: raise ValueError("Unknown query flag '%s'" % tok)
			else:
				# Normal field. Perform a few convenience transformations first.
				if tok.startswith("@@"):
					# Hack. *Force* the given ids to be returned, even if they aren't in the database.
					override_ids = load_ids(tok[2:])
					continue
				elif tok.startswith("@"):
					# Restrict dataset to those in the given file
					tok = "file_contains('%s',id)" % tok[1:]
				elif tok.startswith("~@"):
					tok = "~file_contains('%s',id)" % tok[2:]
				fields.append(tok)
		if override_ids is not None:
			return override_ids
		# Apply our default queries here. These are things that we almost always
		# want in our queries, and that it's tedious to have to specify manually
		# each time. For example, this would be "selected" for act todinfo queries
		if apply_default_query:
			fields = fields + utils.split_outside(self.default_query,",")
		# Now evaluate our fields one by one. This is done so that
		# function fields can inspect the current state at that point
		for field in fields:
			scope = np.__dict__.copy()
			scope.update(data)
			for name, functor in self.functors.iteritems():
				scope[name] = functor(data)
			with utils.nowarn():
				hits = eval(field, scope)
			# Restrict all fields to the result
			data = dslice(data, hits)
		# Split the rest into a sorting field and a slice
		toks = rest.split("[")
		if   len(toks) == 1: sort, fsel, dsel = toks[0], "", ""
		elif len(toks) == 2: sort, fsel, dsel = toks[0], "", "["+toks[1]
		else: sort, fsel, dsel = toks[0], "["+toks[1], "["+"[".join(toks[2:])
		if self.sort and not sort: sort = self.sort
		if sort:
			# Evaluate sorting field
			field = data[sort]
			field = eval("field" + fsel)
			data  = dslice(data, np.argsort(field))
		# Finally apply the data slice
		inds = np.arange(len(data["id"]))
		inds = eval("inds" + dsel)
		data = dslice(data, inds)
		return data["id"]
	def __add__(self, other):
		"""Produce a new tagdb which contains the union of the
		tag info from each."""
		res = self.copy()
		res.data = merge([self.data,other.data])
		return res
	def write(self, fname, type=None):
		write(fname, self, type=type)
	@classmethod
	def read(cls, fname, type=None, vars={}):
		"""Read a Tagdb from in either the hdf or text format. This is
		chosen automatically based on the file extension."""
		if type is None:
			if fname.endswith(".hdf"): type = "hdf"
			else: type = "txt"
		if type == "txt":   return cls.read_txt(fname, vars=vars)
		elif type == "hdf": return cls.read_hdf(fname)
		else: raise ValueError("Unknown Tagdb file type: %s" % fname)
	@classmethod
	def read_txt(cls, fname, vars={}):
		"""Read a Tagdb from text files. Only supports boolean tags."""
		datas = []
		for subfile, tags in parse_tagfile_top(fname, vars=vars):
			ids = parse_tagfile_idlist(subfile)
			data = {"id":ids}
			for tag in tags:
				data[tag] = np.full(len(ids), True, dtype=bool)
			datas.append(data)
		return cls(merge(datas))
	@classmethod
	def read_hdf(cls, fname):
		"""Read a Tagdb from an hdf file."""
		data = {}
		with h5py.File(fname, "r") as hfile:
			for key in hfile:
				data[key] = hfile[key].value
		return cls(data)
	def write(self, fname, type=None):
		"""Write a Tagdb in either the hdf or text format. This is
		chosen automatically based on the file extension."""
		if type is None:
			if fname.endswith(".hdf"): type = "hdf"
			else: type = "txt"
		if type == "txt":   raise NotImplementedError
		elif type == "hdf": return self.write_hdf(fname)
		else: raise ValueError("Unknown Tagdb file type: %s" % fname)
	def write_hdf(self, fname):
		"""Write a Tagdb to an hdf file."""
		with h5py.File(fname, "w") as hfile:
			for key in self.data:
				hfile[key] = self.data[key]

def dslice(data, inds):
	return {key:val[...,inds] for key, val in data.iteritems()}

# We want a way to build a dtype from file. Two main ways will be handy:
# 1: The tag fileset.
#    Consists of a main file with lines like
#    filename tag tag tag ...
#    where each named file contains one id per line
#    (though in practice there may be other stuff on the lines that needs cleaning...)
# 2: An hdf file

def read(fname, type=None, vars={}): return Tagdb.read(fname, type=type, vars=vars)
def read_txt(fname, vars={}): return Tagdb.read_txt(fname, vars=vars)
def read_hdf(fname): return Tagdb.read_hdf(fname)

def write(fname, tagdb, type=None): return tagdb.write(fname, type=type)
def write_hdf(fname, tagdb): return tagdb.write(fname)

def merge(tagdatas):
	"""Merge two or more tagdbs into a total one, which will have the
	union of the ids."""
	# First get rid of empty inputs
	tagdatas = [data.copy() for data in tagdatas if len(data["id"]) > 0]
	# Generate the union of ids, and the index of each
	# tagset into it.
	tot_ids = utils.union([data["id"] for data in tagdatas])
	inds = [utils.find(tot_ids, data["id"]) for data in tagdatas]
	for data in tagdatas: data["id"] = data["id"].astype(tot_ids.dtype)
	nid  = len(tot_ids)
	data_tot = {}
	for di, data in enumerate(tagdatas):
		for key, val in data.iteritems():
			if key not in data_tot:
				# Hard to find an appropriate default value for
				# all types. We use false for bool to let tags
				# accumulate, -1 as probably the most common
				# placeholder value for ints, and NaN for strings
				# and floats.
				oval = np.zeros(val.shape[:-1]+(nid,),val.dtype)
				if oval.dtype == bool:
					oval[:] = False
				elif np.issubdtype(oval.dtype, np.integer):
					oval[:] = -1
				else:
					oval[:] = np.NaN
				if oval.dtype == bool: oval[:] = False
				data_tot[key] = oval
			# Boolean flags combine OR-wise, to let us mention the same
			# id in multiple files
			if val.dtype == bool: data_tot[key][...,inds[di]] |= val
			else: data_tot[key][...,inds[di]] = val
	return data_tot

def parse_tagfile_top(fname, vars={}):
	"""Read and parse the top-level tagfile in the Tagdb text format.
	Contains lines of the form [filename tag tag tag ...]. Also supports
	comments (#) and variables (foo = bar), which can be referred to later
	as {foo}. Returns a list of (fname, tagset) tuples."""
	res  = []
	vars = dict(vars)
	with open(fname,"r") as f:
		for line in f:
			line = line.rstrip()
			if not line or len(line) < 1 or line[0] == "#": continue
			toks = shlex.split(line)
			assert len(toks) > 1, "Tagdb entry needs at least one tag: '%s'" % line
			if toks[1] == "=":
				vars[toks[0]] = toks[2]
			else:
				res.append((toks[0].format(**vars), set(toks[1:])))
	return res

def parse_tagfile_idlist(fname):
	"""Reads a file containing an id per line, and returns the ids as a list."""
	res = []
	with open(fname,"r") as f:
		for line in f:
			line = line.rstrip()
			if len(line) < 1 or line[0] == "#": continue
			res.append(line.split()[0])
	return res

def file_contains(fname, ids):
	lines = [line.split()[0] for line in open(fname,"r") if not line.startswith("#")]
	return utils.contains(ids, lines)

def load_ids(fname):
	lines = [line.split()[0] for line in open(fname,"r") if not line.startswith("#")]
	return np.array(lines)

def split_ids(ids):
	bids, subids = [], []
	for id in ids:
		toks = id.split(":")
		bids.append(toks[0])
		subids.append(toks[1] if len(toks) > 1 else "")
	return bids, subids

def merge_subid(a, b):
	res = set(a.split(",")) | set(b.split(","))
	try: res.remove("")
	except: pass
	return ",".join(sorted(list(res)))

def append_subs(ids, subs):
	if len(ids) == 0: return ids
	sep_helper = np.array(["",":"])
	ind = (np.char.str_len(subs) > 0).astype(int)
	sep = sep_helper[ind]
	return np.char.add(ids, np.char.add(sep, subs))
