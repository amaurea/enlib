include ../compile_opts/$(ENLIB_COMP).mk

fortran.so: fortran.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) -c -m fortran{,.f90} $(LAPACK_LINK) $(OMP_LINK)
	#F90FLAGS="-fopenmp -Ofast -ffree-line-length-none" f2py2 -c -m fortran{,.f90} -lgomp -llapack -lblas
	#F90FLAGS="-fopenmp -O0 -fbounds-check -ffree-line-length-none" f2py2 -c -m fortran{,.f90} -lgomp -llapack -lblas
	#F90FLAGS="-fopenmp -O3" f2py --fcompiler=intelem -c -m fortran{,.f90} -L$(MKLPATH)  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -liomp5

clean:
	rm -rf *.pyc *.so
