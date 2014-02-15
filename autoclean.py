"""This module defines a class decorator that makes sure that the __exit__
function gets called when the program exits, if it hasn't been called before.
Its purpose is to allow interactive use of resource-using classes. In those
situations, the standard "with" approach does not work."""
import atexit

_toclean_ = set()
def call_exit_for_objects(objs):
	"""Calls __exit__ for all objects in objs, leaving objs empty."""
	# To allow the __exit__ function of the removed objects to
	# clean up other objects that have been registered, we must
	# take into account that __exit__ may modify the toclean list.
	# We therefore use a while loop, and keep handling and removing
	# the first item.
	while len(objs) > 0:
		obj = objs.pop()
		obj.__exit__(None,None,None)
atexit.register(call_exit_for_objects, _toclean_)

def autoclean(cls):
	global _toclean_
	# Fail on purpose if __init__ and __exit__ don't exist.
	oldinit  = cls.__init__
	oldexit  = cls.__exit__
	def newinit(self, *args, **kwargs):
		oldinit(self, *args, **kwargs)
		_toclean_.add(self)
	def newexit(self, type, value, traceback):
		try:    _toclean_.remove(self)
		except KeyError: pass
		oldexit(self, type, value, traceback)
	cls.__init__ = newinit
	cls.__exit__ = newexit
	return cls
