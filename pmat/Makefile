include ../compile_opts/$(ENLIB_COMP).mk

all: pmat_core_32.so pmat_core_64.so

%.so: %.f90
	#$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $< $(OMP_LINK)
	# ifort is a bit slower than gfortran here
	#FFLAGS="-fopenmp -fPIC -Ofast -ffree-line-length-none -Wtabs" $(F2PY) --fcompiler=gfortran -c -m $(basename $<) $< -lgomp
	FFLAGS="-fopenmp -fPIC -O0 -fbounds-check -ffree-line-length-none" $(F2PY) --fcompiler=gfortran -c -m $(basename $<) $< -lgomp

pmat_core_32.f90: pmat_core.F90
	$(SED) 's/\<_\>/4/g' < $< > $@
pmat_core_64.f90: pmat_core.F90
	$(SED) 's/\<_\>/8/g' < $< > $@

clean:
	rm -rf *.pyc *.so
