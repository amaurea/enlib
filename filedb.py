"""This module provides access to the location of a set of related files.
After initialization, it can be queried given an "id", and responds with
an object containing the paths to all the files corresponding to that "id".
For example, for the actpol data set, querying with "1376512459.1376536951.ar1"
would respond with an object giving the location of the TOD, cuts, gains, etc.
for that id."""
import glob, shlex, re, bunch, itertools
class Filedb:
	def __init__(self, desc_file):
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
		self.files   = {}
		self.rules   = {}
		self.ids     = []
		with open(desc_file,"r") as file:
			for line in file:
				toks = shlex.split(line)
				name, globs = toks[0][:-1], toks[1:]
				if name == "id":
					other = globs[0]
					regex = [re.compile(g) for g in globs[1:]]
					for fn in self.files[other][0]:
						for r in regex:
							m = r.search(fn)
							if m:
								self.ids.append(m.group(1))
								break
				else:
					groups = itertools.groupby(toks, lambda x: x == ":")
					files  = [[fn for tok in group for fn in glob.glob(tok)] for k, group in groups if not k]
					self.files[name] = files
					self.rules[name] = globs
	def __getitem__(self, id):
		"""Returns a bunch describing all the paths corresponding to id. None is returned
		for entries with no matches. A filename is considered to match if it contains id
		as a substring."""
		res = bunch.Bunch()
		for c in self.files:
			files = self.files[c]
			for fileset in files:
				match = next((f for f in fileset if id in f), fileset[0] if len(fileset) == 1 else None)
				print fileset, match
				if match:
					res[c] = match
					break
		res.id = id
		return res
