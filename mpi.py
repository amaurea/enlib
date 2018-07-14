"""Utilities for making mpi use safer and easier."""
import sys,os

try:
	disable_mpi_env = os.environ['DISABLE_MPI']
	disable_mpi = True if disable_mpi_env.lower().strip() == "true" else False
except:
	disable_mpi = False

if not(disable_mpi):
	from mpi4py.MPI import *
# Uncaught exceptions don't cause mpi to abort. This can lead to thousands of
# wasted CPU hours
def cleanup(type, value, traceback):
	sys.__excepthook__(type, value, traceback)
	COMM_WORLD.Abort(1)
sys.excepthook = cleanup
