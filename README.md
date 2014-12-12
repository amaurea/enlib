Dependencies
============
Enlib consists of a collection of relatively independend modules, so
if you don't need all of them you don't need to install all dependencies.
I've split the modules into groups based on typical use cases and list
the dependencies of each.

Misc. utils
-----------
* Primary modules: utils, cg, colors, bins, errors
* Internal dependencies: None
* External dependencies: python2, numpy

Coordinate transformation
-------------------------
* Primary modules: coordinates
* Internal dependencies: iers + [misc utils]
* External dependencies: pyephem, astropy
* To build: make coordinates

Flat-sky maps
-------------
* Primary modules: enmap, powspec
* Internal dependencies: fft, array\_ops, utils, wcs, slice + [misc utils]
* External dependencies: scipy, pyfftw, astropy, h5py
* To build: make array\_ops

Curved-sky maps
---------------
* Primary modules: curvedsky, lensing
* Internal dependencies: sharp + [flat-sky maps]
* External dependencies: libsharp
* To build: make sharp

Mapmaking
---------
* Primary modules: zgetdata, map\_equation, scan, scansim, gapfill, rangelist
* Internal dependencies: [misc. utils, coordinate transformation, flat-sky maps]
* External dependencies: mpi4py, pygetdata, zipfile, tempfile, shutil, bunch
* To build: make
