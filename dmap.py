"""This module implements mpi-distributed maps, where what is conceptually
a single contiguous enmap is split into tiles, with with tile belonging to
an mpi task.

The map is assumed to correspond to data from a set of scans,
which are represented here as a box per scan, with the scans also belonging
to mpi tasks. It is in general impossible to ensure a one-to-one mapping
between scan groups and tile groups, so the scans a given task owns will not
completely overlap with the tiles it owns. But hopefully the overlap won't be
too bad.

The area covered by the bounding box of all the scans a task owns is its
workspace. Projecting from scans to tiles is in principle:

	for scan in scans: scan -> workspace
	for wi in winfo:
		wtile = workspace[wi.slice]
		send(wi.id, wtile)
	for ti in tinfo:
		wtile = recv(ti.id)
		tiles[ti.i][ti.slice] += wtile

And projecting back:

	for ti in tinfo:
		wtile = tiles[ti.i][ti.slice]
		send(ti.id, wtile)
	for wi in winfo:
		wtile = recv(wi.id)
		workspace[wi.slice] = wtile
	for scan in scans: workspace -> scan

The complication is that sends and receives have to happen at the same time.
The easiest way to do this is via alltoallv, which requires the use of
flattened arrays.
"""
import numpy as np, mpi4py.MPI

class Dmap:
	def __init__(self, shape, wcs, boxes, comm=None):
		if comm is None: comm = mpi4py.MPI.COMM_WORLD
		# 1. Compute local box
		# 2. Set up local workspace
		# 3. Define tiling
		# 4. Define tile ownership
		# 5. Find which tiles intersect my boxes
		# 6. Exchange this information with others
		# 7. Define tile-worspace send slices and receive slices
		# 8. Build alltoallv structures:
		#    a) workspace,winfo <-> data,id_lens
		#    b) mytiles,tinfo <-> data,id_lens
		# 9. Implement alltoallv call
