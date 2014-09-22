include ../compile_opts/$(ENLIB_COMP).mk

nmat_core.so: nmat_core.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) -c -m nmat_core{,.f90} $(LAPACK_LINK) $(OMP_LINK)
	#F90FLAGS="-fopenmp -Ofast -ffree-line-length-none" f2py2 -c -m nmat_core{,.f90} -llapack -lblas -lgomp
	#F90FLAGS="-fopenmp -O0 -fbounds-check -ffree-line-length-none" f2py2 -c -m nmat_core{,.f90} -llapack -lblas -lgomp
	#F90FLAGS="-fopenmp -O3" f2py --fcompiler=intelem -c -m nmat_core{,.f90} -L$(MKLPATH)  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -liomp5
clean:
	rm -rf *.pyc *.so
