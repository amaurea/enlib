include ../compile_opts/$(ENLIB_COMP).mk

all: fortran_32.so fortran_64.so

%.so: %.f90
	# ifort is a bit slower than gfortran here
	FFLAGS="-fopenmp -fPIC -Ofast -ffree-line-length-none" $(F2PY) --fcompiler=gfortran -c -m $(basename $<) $< -lgomp
	#FFLAGS="-fopenmp -fPIC -O0 -fbounds-check -ffree-line-length-none" $(F2PY) --fcompiler=gfortran -c -m $(basename $<) $< -lgomp

fortran_32.f90: fortran.F90
	sed 's/\<_\>/4/g' < $< > $@
fortran_64.f90: fortran.F90
	sed 's/\<_\>/8/g' < $< > $@

clean:
	rm -rf *.pyc *.so
