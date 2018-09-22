# This module takes a set of equatorial pointings and classifies which
# of several fixed or moving targets it belongs to. The highest priority
# target which overlaps with the pointing provided will be used.
import ephem, numpy as np
import utils

class FixedPatch:
	def __init__(self, name, center, size):
		self.name   = name
		self.center = center
		self.size   = size
	def match(self, point, margin=0):
		return self.distance(point) - margin <= 0
	# What is the minimal separation between point
	# the patch?
	def distance(self, point):
		# Normalize pointing, so that it is comparable
		# with our center position. Since the patch is
		# rectangular, we will use a rectangular sense of
		# distance. That is, the distance between A and B
		# is simply |A.x-B.x| + |A.y-B.y|. Negative distance
		# means that we are inside the object.
		diff = np.abs(utils.rewind(point[:,1:]-self.center, 0, 2*np.pi))
		return np.min(np.sum(np.maximum(diff-self.size,0),1))

eph = {name.lower(): ephem.__dict__[name] for name in ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune"]}

class EphemObj:
	def __init__(self, name, size):
		self.name = name
		self.eph  = eph[name.lower()]()
		self.size = size
	def match(self, point, margin=0):
		return self.distance(point) - self.size - margin < 0
	def distance(self, point, exact=True):
		mjd = point[0,0]
		djd = mjd + 2400000.5 - 2415020
		self.eph.compute(djd)
		pos = np.array([float(self.eph.ra), float(self.eph.dec)])
		# We assume that only points close to the object are
		# relevant, so use flat sky approximation
		diff = utils.rewind(point[:,1:]-pos, 0, 2*np.pi)
		return np.min(np.sum(diff**2,1)**0.5)

class TargetDB:
	def __init__(self, fname):
		self.targets = []
		self.pris    = []
		for line in open(fname,"r"):
			if line.isspace() or line[0] == "#": continue
			toks = line.split()
			name = toks[0]
			pri  = float(toks[1])
			kind = toks[2]
			if kind == "fixed":
				pos  = np.array((float(toks[3]),float(toks[4])))*utils.degree
				size = np.array((float(toks[5]),float(toks[6])))*utils.degree
				self.targets.append(FixedPatch(name, pos, size))
			elif kind == "ephem":
				size = float(toks[3])*np.pi/180 if len(toks) > 3 else utils.degree
				self.targets.append(EphemObj(name, size))
			else:
				class UnknownTargetKind: pass
				raise UnknownTargetKind()
			self.pris.append(pri)
	# This find the best matching object. What is
	# returned is the specific target object that
	# matched. These can have various properties,
	# but are guraanteed to have the property .name,
	# which most users will be interested in.
	# Point has shape [nsamp,{mjd,ra,dec}]
	def match(self, point, margin=0):
		matches = []
		for pri, trg in zip(self.pris, self.targets):
			if trg.match(point, margin=margin):
				matches.append([pri,trg])
		if len(matches) == 0: return None
		matches = sorted(matches)
		return matches[-1][1]
	def distance(self, point):
		return [t.distance(point) for t in self.targets]
