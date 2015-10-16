import numpy as np
from copy  import deepcopy
from enlib import utils, errors

# read_foo should return a Data object, which constains lists of Datum objects, which
# contain:
#  * the foo data
#  * dets and samples fields (can be None if not applicable)
#  * dets_orig and samples_orig (like dets and samples, but aren't modified by
#    subsequent slicing)
#  * det_index, sample_index (which index into foo is the det and sample index.
#    used for slicing)
#  * a name (should be a valid python name)
# A Data object is what one will usually interact with. It represents a set of
# unified Datum objects which all have compatible dets and samples.
#  * dict of Datums
#  * dets and samples
#  * transparent access to data of each Datum

class DataField:
	def __init__(self, name="default", data=None, dets=None, samples=None,
			det_index=None, sample_index=None, force_contiguous=False):
		"""Initialize a DataField object, which encapsulates an array of data
		which may have a detector axis at dimension det_index corresponding
		to detectors dets, and a sample axis at dimension sample_index corresponding
		to the sample range samples. If dets is None or samples is None,
		then the corresponding axis is not present. force_contiguous
		causes the data array to be reallocated each time it is sliced in order
		to ensure contiguity."""
		# Make copies of dets and samples, to avoid the caller changing them
		# from the outside
		if dets is not None: dets = np.array(dets)
		if samples is not None: samples = np.array(samples)
		self.force_contiguous = force_contiguous
		self.name    = name
		self.data    = data
		self.dets    = dets
		self.samples = samples
		self.dets_orig = dets
		self.samples_orig = samples
		self.det_index = det_index
		self.sample_index = sample_index
	def copy(self): return deepcopy(self)
	def restrict(self, dets=None, samples=None):
		self.restrict_dets(dets)
		self.restrict_samples(samples)
		return self
	def restrict_dets(self, dets):
		"""Restricts our set of detectors to those in the given list. Afterwards,
		our dets will contain the same elements in the same order as dets, and
		our data will have changed accordingly. An IndexError exception will
		be raised if a detector is missing."""
		if self.dets is None or dets is None: return self
		# Find positions of requested detectors in our detector array
		dets = np.asarray(dets)
		if np.all(dets == self.dets): return self
		inds = np.argsort(self.dets)
		pos  = inds[np.searchsorted(self.dets, dets, sorter=inds)]
		# Did we actually find the right detectors?
		bad = self.dets[pos] != dets
		if np.any(bad):
			raise IndexError("Detectors %s do not exist in DataField %s" % (str(np.where(bad)[0]), self.name))
		# Update our object
		self.dets = self.dets[pos]
		if self.det_index is not None:
			self.data = self.data[(slice(None),)*self.det_index + (pos,)]
		return self
	def restrict_samples(self, samples):
		"""Restricts our sample range to that given. Samples must be a standard
		half-open from,end pair. An IndexError is raised if the requested range
		falls outside the samples available."""
		if self.samples is None or samples is None: return self
		samples = np.asarray(samples)
		if np.all(samples==self.samples): return self
		if samples[0] < self.samples[0] or samples[1] > self.samples[1]:
			raise IndexError("DataField %s samples %s does not contain requested range %s" % (self.name, str(self.samples), str(samples)))
		sel = slice(samples[0]-self.samples[0],samples[1]-self.samples[0])
		# Update our object
		self.samples = samples
		if self.sample_index is not None:
			self.data = self.data[(slice(None),)*self.sample_index + (sel,)]
		return self
	def __setattr__(self, name, val):
		# Enforce contiguous data if requested
		if name is "data" and self.force_contiguous:
			val = np.ascontiguousarray(val)
		if name in ["dets","samples","dets_orig","samples_orig"] and val is not None:
			val = np.asarray(val)
		self.__dict__[name] = val
	@property
	def ndet(self): return len(self.dets) if self.dets else None
	@property
	def nsamp(self): return self.samples[1]-self.samples[0] if self.samples else None
	def data_desc(self):
		try:
			if self.data is None: return ""
			dims = [str(s) for s in self.data.shape]
			if self.det_index is not None: dims[self.det_index] = "d:"+dims[self.det_index]
			if self.sample_index is not None: dims[self.sample_index] = "s:"+dims[self.sample_index]
		except AttributeError:
			# No shape. But may still be slicable object. Construct less informative version
			nmax = max(self.det_index, self.sample_index)
			if nmax is None: return ""
			else:
				dims = [":" for i in range(nmax+1)]
				if self.det_index is not None: dims[self.det_index] = "d:"+str(len(self.dets))
				if self.sample_index is not None: dims[self.sample_index] = "d:"+str(self.samples[1]-self.samples[0])
		return "[%s]" % ",".join(dims)
	def __repr__(self):
		return "DataField(name='%s', dets=%s, samps=%s, data%s)" % (self.name, str(self.dets), str(self.samples), self.data_desc())

class DataSet:
	def __init__(self, datafields):
		# A DataSet by definition has consistent detectors and samples,
		# so ensure that.
		datafield_intersection(datafields)
		self.datafields = {d.name: d for d in datafields}
		self.dets, self.samples = self._calc_detsamps()
	@property
	def names(self): return self.datafields.keys()
	@property
	def ndet(self): return len(self.dets) if self.dets is not None else None
	@property
	def nsamp(self): return self.samples[1]-self.samples[0] if self.samples is not None else None
	def restrict(self, dets=None, samples=None):
		for k in self.datafields:
			self.datafields[k].restrict(dets, samples)
		self.dets, self.samples = self._calc_detsamps()
		return self
	def copy(self): return deepcopy(self)
	def _calc_detsamps(self):
		dets, samples = None, None
		for k in self.datafields:
			d = self.datafields[k]
			if d.dets is not None: dets = d.dets
			if d.samples is not None: samples = d.samples
		return dets, samples
	def __contains__(self, name):
		return name in self.datafields
	def __setattr__(self, name, value):
		if "datafields" in self.__dict__ and name in self.names:
			self.datafields[name].data = value
		else:
			self.__dict__[name] = value
	def __getattr__(self, name):
		if name in self.__dict__["datafields"].keys():
			return self.__dict__["datafields"][name].data
		raise AttributeError
	def __dir__(self): return sorted(self.__dict__.keys() + self.names)
	def __repr__(self):
		descs = ["%s%s" % (name, self.datafields[name].data_desc()) for name in self.datafields]
		return "DataSet([%s])" % (",".join(descs))
	def __add__(self, other):
		return merge([self,other], copy=True)
	def __iadd__(self, other):
		res = merge([self,other.copy()])
		self.datafields = res.datafields
		self.dets, self.samples = self._calc_detsamps()

def merge(datasets, copy=False):
	"""Merge a list of datasets into a single dataset which will contain all
	the datafields of the individual ones. Each dataset must have unique datafields
	in order to avoid collissions. The resulting dataset will contain the intersection
	of the detectors and samples present in the input data fields."""
	def get_datafields(d):
		if isinstance(d, DataField): return [d]
		else: return d.datafields.values()
	return DataSet(datafield_intersection([df for ds in datasets for df in get_datafields(ds)],copy=copy))

def datafield_intersection(datafields, copy=False):
	"""Restrict a set of datafields to their common detectors and samples.
	If copy is True, then the input datafields are not modified, and the
	new datafields are returned."""
	# Find the common set of detectors
	dets_list = [df.dets for df in datafields if df.dets is not None]
	dets = utils.common_vals(dets_list) if len(dets_list) > 0 else None
	# Find the common set of samples
	samples_list = np.array([df.samples for df in datafields if df.samples is not None])
	samples = np.array([np.max(samples_list[:,0]),np.min(samples_list[:,1])]) if len(samples_list) > 0 else None
	# Preserve data if required
	if copy: datafields = [df.copy() for df in datafields]
	# Restrict each dataset to the intersection
	for df in datafields: df.restrict(dets, samples)
	# And return the resulting dataset
	return datafields
