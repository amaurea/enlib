"""Utilities for making mpi use safer and easier."""
import sys
from mpi4py.MPI import *
# Uncaught exceptions don't cause mpi to abort. This can lead to thousands of
# wasted CPU hours
def cleanup(type, value, traceback):
	sys.__excepthook__(type, value, traceback)
	COMM_WORLD.Abort(1)
sys.excepthook = cleanup
