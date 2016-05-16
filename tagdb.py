"""This module provides a class that lets one associate tags and values
with a set of ids, and then select ids based on those tags and values.
For example query("deep56,night,ar2,in(bounds,Moon)") woult return a
set of ids with the tags deep56, night and ar2, and where th Moon[2]
array is in the polygon specified by the bounds [:,2] array."""
import re, numpy as np, h5py, shlex, copy, warnings
from enlib import utils

class Tagdb:
	def __init__(self, data=None, sort="id"):
		"""Most basic constructor. Takes a dictionary
		data[name][...,nid], which must contain the field "id"."""
		if data is None:
			self.data = {"id":np.zeros(0,dtype='S5')}
		else:
			self.data = {key:np.array(val) for key,val in data.iteritems()}
			assert "id" in self.data, "Id field missing"
			if self.data["id"].size == 0: self.data["id"] = np.zeros(0,dtype='S5')
		self.sort  = sort
	def get_funcs(self):
		return {"file_contains": file_contains}
	def copy(self):
		return copy.deepcopy(self)
	@property
	def ids(self): return self.data["id"]
	def __len__(self): return len(self.ids)
	def __getitem__(self, query=""):
		return self.select(self.query(query))
	def select(self, ids):
		"""Return a tagdb which only contains the selected ids."""
		inds = utils.find(self.ids, ids)
		odata = {key:val[...,inds] for key, val in self.data.iteritems()}
		res = self.copy()
		res.data = odata
		return res
	def query(self, query=None):
		"""Query the database. The query takes the form
		tag,tag,tag,...:sort[slice], where all tags must be satisfied for an id to
		be returned. More general syntax is also available. For example,
		(a+b>c)|foo&bar,cow. This follows standard python and numpy syntax,
		except that , is treated as a lower-priority version of &."""
		# First split off any sorting field or slice
		if query is None: query = ""
		toks = utils.split_outside(query,":")
		query, rest = toks[0], ":".join(toks[1:])
		# Split into ,-separated fields. Fields starting with a "+"
		# are taken to be tag markers, and are simply propagated to the
		# resulting ids.
		toks = utils.split_outside(query,",")
		fields, extra = [], []
		for tok in toks:
			if tok.startswith("+"):
				extra.append(tok[1:])
			else:
				# Normal field. Perform a few convenience transformations first.
				if tok in self.data["id"]:
					tok = "id=='%s'" % tok
				elif tok.startswith("@"):
					tok = "file_contains('%s',id)" % tok[1:]
				fields.append(tok)
		# Back to strings. For our query, we want numpy-compatible syntax,
		# with low precedence for the comma stuff.
		query = "(" + ")&(".join(fields) + ")"
		extra = ",".join(extra)
		# Evaluate the query. First build up the scope dict
		scope = np.__dict__.copy()
		scope.update(self.data)
		# Extra functions
		scope.update(self.get_funcs())
		with utils.nowarn():
			hits = eval(query, scope)
		foo = self.ids == '1452716782.1452746181.ar3'
		ids  = self.data["id"][hits]
		# Split the rest into a sorting field and a slice
		toks = rest.split("[")
		if   len(toks) == 1: sort, fsel, dsel = toks[0], "", ""
		elif len(toks) == 2: sort, fsel, dsel = toks[0], "", "["+toks[1]
		else: sort, fsel, dsel = toks[0], "["+toks[1], "["+"[".join(toks[2:])
		if self.sort and not sort: sort = self.sort
		if sort:
			# Evaluate sorting field
			field = self.data[sort][hits]
			field = eval("field" + fsel)
			inds  = np.argsort(field)
			# Apply sort
			ids   = ids[inds]
		# Finally apply the data slice
		ids = eval("ids" + dsel)
		# Append the unknown tags to the ids
		if extra: ids = np.char.add(ids, ":"+extra)
		return ids
	def __add__(self, other):
		"""Produce a new tagdb which contains the union of the
		tag info from each."""
		return merge([self,other])
	def write(self, fname, type=None):
		write(fname, self, type=type)
	@staticmethod
	def read(fname, type=None, matchfun=None):
		return read(fname, type=type, matchfun=matchfun)

# We want a way to build a dtype from file. Two main ways will be handy:
# 1: The tag fileset.
#    Consists of a main file with lines like
#    filename tag tag tag ...
#    where each named file contains one id per line
#    (though in practice there may be other stuff on the lines that needs cleaning...)
# 2: An hdf file

def read(fname, type=None, matchfun=None):
	"""Read a Tagdb from in either the hdf or text format. This is
	chosen automatically based on the file extension."""
	if type is None:
		if fname.endswith(".hdf"): type = "hdf"
		else: type = "txt"
	if type == "txt":   return read_txt(fname, matchfun)
	elif type == "hdf": return read_hdf(fname)
	else: raise ValueError("Unknown Tagdb file type: %s" % fname)

def read_txt(fname, matchfun=None):
	"""Read a Tagdb from text files. Only supports boolean tags."""
	dbs = []
	for subfile, tags in parse_tagfile_top(fname):
		ids = parse_tagfile_idlist(subfile, matchfun)
		data = {"id":ids}
		for tag in tags:
			data[tag] = np.full(len(ids), True, dtype=bool)
		dbs.append(Tagdb(data))
	return merge(dbs)

def read_hdf(fname):
	"""Read a Tagdb from an hdf file."""
	data = {}
	with h5py.File(fname, "r") as hfile:
		for key in hfile:
			data[key] = hfile[key].value
	return Tagdb(data)

def write(fname, tagdb, type=None):
	"""Write a Tagdb in either the hdf or text format. This is
	chosen automatically based on the file extension."""
	if type is None:
		if fname.endswith(".hdf"): type = "hdf"
		else: type = "txt"
	if type == "txt":   raise NotImplementedError
	elif type == "hdf": return write_hdf(fname, tagdb)
	else: raise ValueError("Unknown Tagdb file type: %s" % fname)

def write_hdf(fname, tagdb):
	"""Write a Tagdb to an hdf file."""
	with h5py.File(fname, "w") as hfile:
		for key in tagdb.data:
			hfile[key] = tagdb.data[key]

def merge(tagdbs):
	"""Merge two or more tagdbs into a total one, which will have the
	union of the ids."""
	# Generate the union of ids, and the index of each
	# tagset into it.
	tot_ids = utils.union([db.data["id"] for db in tagdbs])
	inds = [utils.find(tot_ids, db.data["id"]) for db in tagdbs]
	nid  = len(tot_ids)
	data_tot = {}
	for di, db in enumerate(tagdbs):
		for key, val in db.data.iteritems():
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
	return Tagdb(data_tot)

def parse_tagfile_top(fname):
	"""Read and parse the top-level tagfile in the Tagdb text format.
	Contains lines of the form [filename tag tag tag ...]. Also supports
	comments (#) and variables (foo = bar), which can be referred to later
	as {foo}. Returns a list of (fname, tagset) tuples."""
	res  = []
	vars = {}
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

def parse_tagfile_idlist(fname, matchfun=None):
	"""Reads a file containing an id per line, and returns the ids as a list.
	If regex is specified, it should be function returning an id or none (to
	skip)"""
	res = []
	with open(fname,"r") as f:
		for line in f:
			line = line.rstrip()
			if len(line) < 1 or line[0] == "#": continue
			if matchfun is not None:
				id = matchfun(line)
				if id: res.append(id)
			else:
				res.append(line.split()[0])
	return res

def file_contains(fname, ids):
	lines = [line.split()[0] for line in open(fname,"r") if not line.startswith("#")]
	return utils.contains(ids, lines)
