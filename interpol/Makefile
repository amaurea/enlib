include ../compile_opts/$(ENLIB_COMP).mk

all: fortran_32.so fortran_64.so

%.so: %.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK) skip: map_border

fortran_32.f90: fortran.F90
	$(SED) 's/\<_\>/4/g' < $< > $@
fortran_64.f90: fortran.F90
	$(SED) 's/\<_\>/8/g' < $< > $@

clean:
	rm -rf *.pyc *.so *.mod
