include ../compile_opts/$(ENLIB_COMP).mk

all: fortran_32.so fortran_64.so fortran_c64.so fortran_c128.so

%.so: %.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) -c -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK)

fortran_32.f90: fortran.F90
	$(SED) 's/\<T\>/real/g;s/\<_\>/4/g;s/\<C\>/s/g;s/\<L\>/1e-6/g;s/\<L0\>/1e-25/g;s/\<ONE\>/1.0/g;s/\<ZERO\>/0.0/g;s/\<R,//g;s/\<SY\>/sy/g;s/##//g' < $< > $@
fortran_64.f90: fortran.F90
	$(SED) 's/\<T\>/real/g;s/\<_\>/8/g;s/\<C\>/d/g;s/\<L\>/1d-9/g;s/\<L0\>/1d-50/g;s/\<ONE\>/1d0/g;s/\<ZERO\>/0d0/g;s/\<R,//g;s/\<SY\>/sy/g;s/##//g' < $< > $@
fortran_c64.f90: fortran.F90
	$(SED) 's/\<T\>/complex/g;s/\<_\>/4/g;s/\<C\>/c/g;s/\<L\>/1e-6/g;s/\<L0\>/1e-25/g;s/\<ONE\>/(1.0,0.0)/g;s/\<ZERO\>/(0.0,0.0)/g;s/\<R,/rwork,/g;s/\<SY\>/he/g;s/##//g' < $< > $@
fortran_c128.f90: fortran.F90
	$(SED) 's/\<T\>/complex/g;s/\<_\>/8/g;s/\<C\>/z/g;s/\<L\>/1d-9/g;s/\<L0\>/1d-50/g;s/\<ONE\>/(1d0,0d0)/g;s/\<ZERO\>/(0d0,0d0)/g;s/\<R,/rwork,/g;s/\<SY\>/he/g;s/##//g' < $< > $@


clean:
	rm -rf *.pyc *.so
