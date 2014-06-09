all:pmat_core.so
pmat_core.so: pmat_core.f90
	F90FLAGS="-fopenmp -Ofast -ffree-line-length-none" f2py2 -c -m pmat_core{,.f90} -lgomp
	#F90FLAGS="-fopenmp -Og -g -fbounds-check -ffree-line-length-none" f2py2 -c -m pmat_core{,.f90} -lgomp
clean:
	rm -rf *.pyc *.so
