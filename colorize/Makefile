all:fortran.so
fortran.so: fortran.f90
	F90FLAGS="-fopenmp -Ofast -ffree-line-length-none" f2py2 -c -m fortran{,.f90} -lgomp
clean:
	rm -rf *.pyc *.so
