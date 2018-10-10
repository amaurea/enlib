from . import bunch

class ExecDB:
	"""ExecDB is a flexible way of mapping from dataset ids to
	the files needed for analysing that dataset. It is based on two
	inputs:
	
	1. The database file containes python code that should define
	variables containing the information necessary to load and interpret
	the dataset files. The variables available to the code in this
	file is setup by a second file - the vars setup file.
	2. The vars setup is a file containing python code that will be
	called with the variable "id" defined. Its job is to define variables
	that can be used in the data file. The idea is that this file is
	needs to chance much less often than the other one.

	For example, if vars is 'a = len(id)' and db is
	'if a[0] > 2: moo = id
	 else: moo = id[::-1]'
	then a query with id = "hello" will result in
	{'moo': 'hello', 'id'='hello', 'a': 4}"""
	def __init__(self, db_file=None, vars_file=None, db_data=None, vars_data=None,
			override=None, root=None):
		self.db_source   = read_data(db_file,    db_data)
		if override is not None:
			self.db_source += "\n" + expand_override(override)
		self.vars_source = read_data(vars_file, vars_data, "")
		if root is not None: # Allow relative file names
			self.vars_source = """root = "%s"\n""" % root + self.vars_source
		if self.db_source is None: raise ValueError("No database specified in ExecDB")
		self.db_code    = compile(self.db_source,   "<exec_db,db_source>",   "exec")
		self.vars_code  = compile(self.vars_source, "<exec_db,vars_source>", "exec")
	def __getitem__(self, id): return self.query(id)
	def query(self, id):
		if not isinstance(id, basestring):
			return [self.query(i) for i in id]
		globs, locs = {"id":id}, {}
		exec(self.vars_code, {}, globs)
		exec(self.db_code, globs, locs)
		globs.update(locs)
		locs = recursive_format(locs, globs)
		for key in globs["export"]:
			locs[key] = globs[key]
		return bunch.Bunch(locs)
	def dump(self):
		return self.db_source

def read_data(file_or_fname=None, data=None, default=None):
	"""Helper function for ExecDB. Gets a string of data
	from either a file or the provided data argument"""
	if data is not None: return data
	if file_or_fname is None: return default
	if isinstance(file_or_fname, file):
		return file.read()
	else:
		with open(file_or_fname) as f:
			return f.read()

def recursive_format(data, formats):
	"""Expand all strings contained in dicts, lists or tuples in data
	using string.format with the given formats dict"""
	if isinstance(data, dict):
		data = {key: recursive_format(data[key], formats) for key in data}
	elif isinstance(data, list):
		data = [recursive_format(val, formats) for val in data]
	elif isinstance(data, tuple):
		data = tuple([recursive_format(val, formats) for val in data])
	elif isinstance(data, basestring):
		return data.format(**formats)
	return data

def expand_override(desc):
	lines  = desc.split(",")
	olines = []
	for line in lines:
		toks  = line.split(":")
		olines.append('%s = "%s"' % (toks[0],toks[1]))
	return "\n".join(olines)
