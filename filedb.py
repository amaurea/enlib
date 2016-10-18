"""This module provides access to the location of a set of related files.
After initialization, it can be queried given an "id", and responds with
an object containing the paths to all the files corresponding to that "id".
For example, for the actpol data set, querying with "1376512459.1376536951.ar1"
would respond with an object giving the location of the TOD, cuts, gains, etc.
for that id."""
import glob, shlex, pipes, re, itertools
from enlib import bunch

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

def pre_split(line):
	toks = line.strip().split(":")
	return [toks[0]] + shlex.split(":".join(toks[1:]))

class FormatDB(Basedb):
	"""This File DB variant takes a more general approach than Regdb.
	It does not attempt to describe everything in the parameter file.
	Instead, it is meant to be subclassed or otherwise configured to
	add a set of functions that build strings from the identifiers.
	These are then inserted into the format strings listed in the
	parameter file using string.format. This should allow, compact,
	readable and flexible parameter files."""
	def __init__(self, file=None, data=None, funcs={"id":lambda id:id}, override=None):
		self.funcs = funcs.items()
		self.override = override
		Basedb.__init__(self, file=file, data=data)
	def load(self, data, funcs={}):
		self.rules = []
		self.static = bunch.Bunch()
		for line in data.splitlines():
			line = line.strip()
			if len(line) < 1 or line[0] == "#": continue
			# Split into part before first : and the rest
			toks = pre_split(line)
			if len(toks) == 1: toks = toks + [""]
			# There may be multiple formats on the same line, pipe-separated
			name, format  = toks[0], toks[1:]
			self.rules.append({"name":name, "format": format})
			self.static[name] = format
	def __getitem__(self, id):
		return self.query(id)
	def query(self, id, multi=True):
		# Split the id argument into the actual id plus a tag list
		toks    = id.split(":")
		id  = toks[0]
		tag = toks[1] if len(toks) > 1 else None
		info = {name: fun(id) for name, fun in self.funcs}
		res = bunch.Bunch()
		selected=[True]
		for rule in self.rules:
			name, format = rule["name"], rule["format"]
			if name[0] == "@":
				# In this case, format is actually the conditional in the selector
				if name == "@end":
					selected.pop()
				elif name == "@else":
					selected[-1] = not selected[-1]
				else:
					# General format @var:case case case ...
					match = False
					for case in format:
						match |= ("{%s}"%name[1:]).format(**info) == case
					selected.append(match)
					#selected.append(("{%s}"%name[1:]).format(**info) == format[0])
			elif len(format) == 0 or len(format[0]) == 0:
				# Handle variable assignment. Avoids repeating the same long paths over and over again
				vname, vval = re.split(r"\s*=\s*", name)
				info[vname] = vval
			elif all(selected):
				tmp = [fmt.format(**info) for fmt in rule["format"]]
				res[rule["name"]] = tmp if multi else tmp[0]
		res.id  = id
		res.tag = tag
		# Apply override if specified:
		if self.override and self.override != "none":
			for tok in self.override.split(","):
				name, val = tok.split(":")
				val = val.format(**info)
				res[name] = [val] if multi else val
		return res
	def dump(self):
		lines = []
		for rule in self.rules:
			line = "%s: %s" %(rule["name"], " ".join([pipes.quote(fmt) for fmt in rule["format"]]))
			lines.append(line)
		return "\n".join(lines)
