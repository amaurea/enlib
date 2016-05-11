"""This module provides a class that lets one associate tags and values
with a set of ids, and then select ids based on those tags and values.
For example query("deep56,night,ar2,in(bounds,Moon)") woult return a
set of ids with the tags deep56, night and ar2, and where th Moon[2]
array is in the polygon specified by the bounds [:,2] array."""
import re, numpy as np
from enlib import utils

class Tagdb:
	def __init__(self, ids, data):
		"""Most basic constructor. Takes a list of ids[nid], and a dictionary
		data[name][...,nid]."""
		self.ids  = np.array(ids)
		self.data = {key:np.array(val) for key,val in data.iteritems()}
	def query(self, query):
		"""Query the database. The query takes the form
		tag,tag,tag,..., where all tags must be satisfied for an id to
		be returned. More general syntax is also available. For example,
		(a+b>c)|foo&bar,cow. This follows standard python and numpy syntax,
		except that , is treated as a lower-priority version of &."""
		# Translate a,b,c into (a)&(b)&(c)
		query = "(" + ")&(".join(query.split(",")) + ")"
		# Evaluate the query. First build up the scope dict
		scope = np.__dict__.copy()
		scope.update(self.data)
		# Generate virtual id tags
		scope.update(build_id_tags(query, self.ids))
		# Extra functions
		scope.update({
			"hits": utils.point_in_polygon,
			})
		hits = eval(query, scope)
		return self.ids[hits]

def build_id_tags(query, ids):
	"""Given a query, look for ids in that query, and
	return a dictionary of tags matching those ids,
	each selecting only the entry with that id."""
	id_tags = {}
	for field in re.findall(r"[\w\.]+",query):
		if field in ids:
			id_tags[field] = field == ids
	return id_tags
