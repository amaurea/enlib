"""This module provides access to the location of a set of related files.
After initialization, it can be queried given an "id", and responds with
an object containing the paths to all the files corresponding to that "id".
For example, for the actpol data set, querying with "1376512459.1376536951.ar1"
would respond with an object giving the location of the TOD, cuts, gains, etc.
for that id."""
import glob, shlex, pipes, re, bunch, itertools

class Basedb:
	def __init__(self, file=None, data=None):
		if file != None:
			try:
				with open(file,"r") as fileobj:
					self.load(fileobj.read())
			except AttributeError:
				self.load(file.read())
		else:
			self.load(data)
	def load(self, data):
		raise NotImplementedError
	def dump(self, data):
		raise NotImplementedError
	def __getitem__(self, id):
		raise NotImplementedError

class Filedb(Basedb):
	"""Initializes the database based on a simple file with lines with
	format name: glob-path [glob-path] [: glob-path ...], which can be quoted.
	For each name specified this way, there will be an entry in the
	"entries" returned from a query with the matching filename from
	that category. The right hand side consists of a " : "-separated
	list of globs. The colons separate priority classes - in case
	no match is found in one, the next list is searched, and so on,
	until a match is found. If a list only contains a single file, that will
	be returned regardless of matching to allow for files that are
	shared for all ids. Hence, a list of specific files followed by
	a default can be specified as
		foo: foofiles/*.txt : foodefault.txt
	
	The special class id specifies a regular expression to apply
	to another class in order to recover ids. This allows Filedb to
	not only passively supply paths in response to ids, but also to
	supply a list of ids to begin with (though no validation is done
	to ensure that files exist for all those ids). The format for
	this rule is: id: [name] [regex]. The specified class must
	already have been mentioned in the file.
	"""
	def load(self, data):
		self.files   = {}
		self.rules   = {}
		self.ids     = []
		for line in data.splitlines():
			# Shlex split is 100 times slower than string split, and is only
			# needed if our line contains quotes
			toks = shlex.split(line) if "'" in line or '"' in line or '\\' in line else line.split()
			name, globs = toks[0][:-1], toks[1:]
			if name == "id":
				other = globs[0]
				regex = [re.compile(g) for g in globs[1:]]
				self.idrule = globs
				for fn in self.files[other][0]:
					for r in regex:
						m = r.search(fn)
						if m:
							self.ids.append(m.group(1))
							break
			else:
				groups = [list(group) for k, group in itertools.groupby(globs, lambda x: x == ":") if not k]
				files  = [[fn for tok in group for fn in cheap_glob(tok)] for group in groups]
				self.files[name] = files
				self.rules[name] = groups
	def dump(self, expand=False):
		lines = []
		if expand:
			for name in self.files:
				lines.append("%s: %s" % (name, " : ".join([" ".join([pipes.quote(r) for r in group]) for group in self.files[name]])))
		else:
			for name in self.rules:
				lines.append("%s: %s" % (name, " : ".join([" ".join([pipes.quote(r) for r in group]) for group in self.rules[name]])))
		lines.append('id: %s' % (" ".join([pipes.quote(r) for r in self.idrule])))
		return "\n".join(lines)

	def __getitem__(self, id):
		"""Returns a bunch describing all the paths corresponding to id. None is returned
		for entries with no matches. A filename is considered to match if it contains id
		as a substring."""
		res = bunch.Bunch()
		for c in self.files:
			files = self.files[c]
			for fileset in files:
				match = next((f for f in fileset if id in f), fileset[0] if len(fileset) == 1 else None)
				if match != None:
					res[c] = match
					break
			else:
				res[c] = None
		res.id = id
		return res

def cheap_glob(fname):
	return glob.glob(fname) if re.search("[][*?]", fname) else [fname]

class Regdb(Basedb):
	"""Filedb needs to search through the file system to identify the files,
	which can be very slow on some systems. This alternative class computes the
	location of the resource files based on regular expressions without ever
	looking at the file system. The cost is that it can't be queried for a list
	of valid ids, which must be provided from the outside. It is also limited
	to the sort of substitutions that can be expressed as a regular expression."""
	def load(self, data):
		"""Each line that does not begin with # and isn't blank is a rule of
		the form name: substitution [regex], where regex defaults to "(.*)"""
		self.rules = []
		for line in data.splitlines():
			if len(line) < 1 or line[0] == "#": continue
			toks = shlex.split(line)
			assert len(toks)>=2
			name  = toks[0][:-1]
			subst = toks[1]
			regex = toks[2] if len(toks)>2 else r"(.*)"
			self.rules.append({"name":name, "regex": re.compile(regex), "subst": subst})
	def __getitem__(self, id):
		"""Returns a bunch describing all the paths corresponding to id."""
		res = bunch.Bunch()
		for rule in self.rules:
			res[rule["name"]] = rule["regex"].sub(rule["subst"], id)
		res.id = id
		return res
	def dump(self):
		lines = []
		for rule in self.rules:
			line = "%s: %s" %(rule["name"], pipes.quote(rule["subst"]))
			if rule["regex"].pattern != r"(.*)":
				line += " " + pipes.quote(rule["regex"].pattern)
			lines.append(line)
		return "\n".join(lines)
