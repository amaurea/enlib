"""This module provides a class that lets one associate tags and values
with a set of ids, and then select ids based on those tags and values.
For example query("deep56,night,ar2,in(bounds,Moon)") woult return a
set of ids with the tags deep56, night and ar2, and where th Moon[2]
array is in the polygon specified by the bounds [:,2] array."""
import re, numpy as np, h5py, shlex
from enlib import utils

class Tagdb:
	def __init__(self, data):
		"""Most basic constructor. Takes a dictionary
		data[name][...,nid], which must contain the field "id"."""
		self.data = {key:np.array(val) for key,val in data.iteritems()}
	def query(self, query):
		"""Query the database. The query takes the form
		tag,tag,tag,..., where all tags must be satisfied for an id to
		be returned. More general syntax is also available. For example,
		(a+b>c)|foo&bar,cow. This follows standard python and numpy syntax,
		except that , is treated as a lower-priority version of &."""
		# Translate a,b,c into (a)&(b)&(c)
		query = "(" + ")&(".join(utils.split_outside(query,",")) + ")"
		# Evaluate the query. First build up the scope dict
		scope = np.__dict__.copy()
		scope.update(self.data)
		# Generate virtual id tags
		scope.update(build_id_tags(query, self.data["id"]))
		# Extra functions
		scope.update({
			"hits": utils.point_in_polygon,
			})
		hits = eval(query, scope)
		return self.data["id"][hits]
	def __add__(self, other):
		"""Produce a new tagdb which contains the union of the
		tag info from each."""
		return merge([self,other])

# We want a way to build a dtype from file. Two main ways will be handy:
# 1: The tag fileset.
#    Consists of a main file with lines like
#    filename tag tag tag ...
#    where each named file contains one id per line
#    (though in practice there may be other stuff on the lines that needs cleaning...)
# 2: An hdf file

def read(fname, type=None, regex=None):
	"""Read a Tagdb from in either the hdf or text format. This is
	chosen automatically based on the file extension."""
	if type is None:
		if fname.endswith(".hdf"): type = "hdf"
		else: type = "txt"
	if type == "txt":   return read_txt(fname, regex)
	elif type == "hdf": return read_hdf(fname)
	else: raise ValueError("Unknown Tagdb file type: %s" % fname)

def read_txt(fname, regex=None):
	"""Read a Tagdb from text files. Only supports boolean tags."""
	dbs = []
	for subfile, tags in parse_tagfile_top(fname):
		ids = parse_tagfile_idlist(subfile, regex)
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

def build_id_tags(query, ids):
	"""Given a query, look for ids in that query, and
	return a dictionary of tags matching those ids,
	each selecting only the entry with that id."""
	id_tags = {}
	for field in re.findall(r"[\w\.]+",query):
		if field in ids:
			id_tags[field] = field == ids
	return id_tags

def merge(tagdbs, default=np.NaN, typewise={bool:False, int:-1}):
	"""Merge two or more tagdbs into a total one, which will have the
	union of the ids. Tags that are missing from some of them will
	be filled with a default value."""
	# Generate the union of ids, and the index of each
	# tagset into it.
	tot_ids = utils.union([db.data["id"] for db in tagdbs])
	inds = [utils.find(tot_ids, db.data["id"]) for db in tagdbs]
	nid  = len(tot_ids)
	data_tot = {}
	for di, db in enumerate(tagdbs):
		for key, val in db.data.iteritems():
			if key not in data_tot:
				# This results in default values of False for bool,
				# NaN for floats, "nan" for strings and -lots for int.
				oval = np.zeros(val.shape[:-1]+(nid,),val.dtype)
				oval[:] = default
				for type_category in typewise:
					if np.issubdtype(val.dtype, type_category):
						oval[:] = typewise[type_category]
				data_tot[key] = oval
			data_tot[key][...,inds[di]] = val
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

def parse_tagfile_idlist(fname, regex=None):
	"""Reads a file containing an id per line, and returns the ids as a list.
	If regex is specified, it should be a capturing regular expression whose
	first capture is to be used as the id."""
	res = []
	with open(fname,"r") as f:
		for line in f:
			line = line.rstrip()
			if len(line) < 1 or line[0] == "#": continue
			if regex is not None:
				m = re.match(regex, line)
				if m: line = m.group(1)
			else:
				line = line.split()[0]
			res.append(line)
	return res
