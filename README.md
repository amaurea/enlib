Quick Install
=============

Very basic but critical functionality can be obtained by following
these steps:

- `git clone` this repo into your directory of choice `$DIR`
(Note: this means `enlib` will be in `$DIR/enlib`. For e.g.
the path to this file will be `$DIR/enlib/README.md`)
- add `$DIR` to your `$PYTHONPATH`

This should allow you to import and use many `enlib` modules
assuming that basic dependencies like `numpy` are present on 
your system.

Dependencies
============
Enlib consists of a collection of relatively independend modules, so
if you don't need all of them you don't need to install all dependencies.
I've split the modules into groups based on typical use cases and list
the dependencies of each. Some modules require compiling via `f2py`.
This requires a fortran and C compiler, which settings are specified
in the `compile_opts` directory. Which settings file is used is specified
via the `ENLIB_COMP` environment variable. You will probably want to make
your own file with settings for your computer, for example `foo.mk`, and
then `export ENLIB_COMP=foo` in your `.bashrc` or similar.

Misc. utils
-----------
* Primary modules: `utils`, `cg`, `colors`, `bins`, `errors`
* Internal dependencies: None
* External dependencies: `python2`, `numpy`

Coordinate transformation
-------------------------
* Primary modules: `coordinates`
* Internal dependencies: `iers` + [misc utils]
* External dependencies: `pyephem`, `astropy`
* To build: `make coordinates`

Flat-sky maps
-------------
* Primary modules: `enmap`, `powspec`
* Internal dependencies: `fft`, `array_ops`, `utils`, `wcs`, `slice` + [misc utils]
* External dependencies: `scipy`, `pyfftw`, `astropy`, `h5py`, `lapack`
* To build: `make array_ops`

Curved-sky maps
---------------
* Primary modules: `curvedsky`, `lensing`
* Internal dependencies: `sharp` + [flat-sky maps]
* External dependencies: [`libsharp`](http://sourceforge.net/projects/libsharp/), [`cython`](http://cython.org)
* To build: `make sharp`

Mapmaking
---------
* Primary modules: `zgetdata`, `map_equation`, `scan`, `scansim`, `gapfill`, `rangelist`
* Internal dependencies: [misc. utils, coordinate transformation, flat-sky maps]
* External dependencies: `mpi4py`, `pygetdata`, `zipfile`, `tempfile`, `shutil`
* To build: `make`

Issues
======

* gfortran has gotten stricter/stupider about argument mismatches, which makes it difficult
  to apply blas to sub-sections of arrays without copying. This is used in the enlib noise
  matrix acceleration, so if you're trying to compile enlib with a recent gfortran you
  might get an error like "Type mismatch between actual argument at (1) and actual argument at (2)"
  or "Element of assumed-shape or pointer array as actual argument at (1) cannot correspond to actual argument at (2)".
  Pass -fallow-argument-mismatch to fix this. This can be set in your compile\_opts.
