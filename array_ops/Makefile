include ../compile_opts/$(ENLIB_COMP).mk

all: fortran_32.so fortran_64.so fortran_c64.so fortran_c128.so

%.so: %.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK)

fortran_32.f90: fortran.F90
	perl -pe 's/\bT\b/real/g;s/\b_\b/4/g;s/\bC\b/s/g;s/\bONE\b/1.0/g;s/\bZERO\b/0.0/g;s/\bR,//g;s/\bSY\b/sy/g;s/##//g' < $< > $@
fortran_64.f90: fortran.F90
	perl -pe 's/\bT\b/real/g;s/\b_\b/8/g;s/\bC\b/d/g;s/\bONE\b/1d0/g;s/\bZERO\b/0d0/g;s/\bR,//g;s/\bSY\b/sy/g;s/##//g' < $< > $@
fortran_c64.f90: fortran.F90
	perl -pe 's/\bT\b/complex/g;s/\b_\b/4/g;s/\bC\b/c/g;s/\bONE\b/(1.0,0.0)/g;s/\bZERO\b/(0.0,0.0)/g;s/\bR,/rwork,/g;s/\bSY\b/he/g;s/##//g' < $< > $@
fortran_c128.f90: fortran.F90
	perl -pe 's/\bT\b/complex/g;s/\b_\b/8/g;s/\bC\b/z/g;s/\bONE\b/(1d0,0d0)/g;s/\bZERO\b/(0d0,0d0)/g;s/\bR,/rwork,/g;s/\bSY\b/he/g;s/##//g' < $< > $@


clean:
	rm -rf *.pyc *.so
