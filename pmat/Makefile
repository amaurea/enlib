include ../compile_opts/$(ENLIB_COMP).mk

all: pmat_core_32.so pmat_core_64.so

%.so: %.f90
	# ifort is a bit slower than gfortran here
	FFLAGS="-fopenmp -fPIC -Ofast -ffree-line-length-none" $(F2PY) --fcompiler=gfortran -c -m $(basename $<) $< -lgomp

pmat_core_32.f90: pmat_core.F90
	sed 's/\<_\>/4/g' < $< > $@
pmat_core_64.f90: pmat_core.F90
	sed 's/\<_\>/8/g' < $< > $@

clean:
	rm -rf *.pyc *.so
