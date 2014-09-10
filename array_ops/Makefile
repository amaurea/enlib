all:fortran.so
fortran.so: fortran.f90
	#F90FLAGS="-fopenmp -Ofast -ffree-line-length-none" f2py2 -c -m fortran{,.f90} -lgomp -llapack -lblas
	#F90FLAGS="-fopenmp -O0 -fbounds-check -ffree-line-length-none" f2py2 -c -m fortran{,.f90} -lgomp -llapack -lblas
	F90FLAGS="-fopenmp -O3" f2py --f90exec=/scinet/gpc/intel/ics/composer_xe_2011_sp1.9.293/bin/intel64/ifort --f77exec=/scinet/gpc/intel/ics/composer_xe_2011_sp1.9.293/bin/intel64/ifort -c -m fortran{,.f90} -L$(MKLPATH)  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -liomp5
clean:
	rm -rf *.pyc *.so
