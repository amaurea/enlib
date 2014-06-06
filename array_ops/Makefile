all:fortran.so
fortran.so: fortran.f90
	F90FLAGS="-fopenmp -Ofast -ffree-line-length-none" f2py2 -c -m fortran{,.f90} -lgomp -llapack -lblas
	#F90FLAGS="-fopenmp -O0 -fbounds-check -ffree-line-length-none" f2py2 -c -m fortran{,.f90} -lgomp -llapack -lblas
clean:
	rm -rf *.pyc *.so
