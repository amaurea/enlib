nmat_core.so: nmat_core.f90
	F90FLAGS="-fopenmp -Ofast -ffree-line-length-none" f2py2 -c -m nmat_core{,.f90} -llapack -lblas -lgomp
	#F90FLAGS="-fopenmp -O0 -fbounds-check -ffree-line-length-none" f2py2 -c -m nmat_core{,.f90} -llapack -lblas -lgomp
clean:
	rm -rf *.pyc *.so
