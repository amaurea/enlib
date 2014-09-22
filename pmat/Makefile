include ../compile_opts/$(ENLIB_COMP).mk
pmat_core.so: pmat_core.f90
	# ifort is a bit slower than gfortran here
	F90FLAGS="-fopenmp -Ofast -ffree-line-length-none" $(F2PY) --fcompiler=gfortran -c -m pmat_core{,.f90} -lgomp
	#$(F2PY) --fcompiler=$(F2PYCOMP) -c -m pmat_core{,.f90} $(OMP_LINK)
	#F90FLAGS="-fopenmp -O0 -g -fbounds-check -ffree-line-length-none" f2py2 -c -m pmat_core{,.f90} -lgomp
clean:
	rm -rf *.pyc *.so
