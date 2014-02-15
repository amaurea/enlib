"""This module is intended as a drop-in replacement for pygetdata. It extends it with
transparent support for zip-compressed. It contains the whole pygetdata namespace, but
replaces the dirfile function."""
from pygetdata import *
from enlib.autoclean import autoclean
import os, tempfile, shutil, zipfile

orig_dirfile = dirfile

@autoclean
class dirfile:
	"""This is a wrapper for pygetdata.dirfile. It extends its functionality
	to handle zip-compressed dirfiles. It supports all the original members,
	even though tab completion in ipython might not indicate so."""
	#Because one cannot inherit from dirfile, this class instead forwards
	#everything manually.
	def __init__(self, name, flags=IGNORE_DUPS, sethandler=None, extra=None):
		"""Open a drifile with file name "name". If this ends with ".zip", it
		Will be assumed to be a zip-copmressed dirfile. Otherwise, it is handled
		normally."""
		if name[-4:] == ".zip":
			with zipfile.ZipFile(name) as z: format  = z.read("format")
			archive = os.path.abspath(name)[:-4]
			self.tmpdir = tempfile.mkdtemp('zdirfile')
			with open(self.tmpdir + "/format","w") as fmtfile:
				fmtfile.write("/ENCODING zzslim %s\n" % archive)
				fmtfile.write(format)
			self.dfile = orig_dirfile(self.tmpdir, flags, sethandler, extra)
		else:
			self.tmpdir = None
			self.dfile = orig_dirfile(name, flags, sethandler, extra)
	def close(self):
		"""Closes the dirfile, and cleans up any resources used."""
		self.dfile.close()
		if self.tmpdir != None:
			try: shutil.rmtree(self.tmpdir)
			except OSError: pass
	def __enter__(self):
		return self
	def __exit__(self, type, value, traceback):
		self.close()
	def __getattr__(self, item):
		return getattr(self.dfile, item)
