include ../compile_opts/$(ENLIB_COMP).mk

all: nmat_core_32.so nmat_core_64.so

%.so: %.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) -c -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK)

nmat_core_32.f90: nmat_core.F90
	sed 's/\<_\>/4/g;s/\<C\>/c/g;s/\<E\>/e/g;s/##//g' < $< > $@
nmat_core_64.f90: nmat_core.F90
	sed 's/\<_\>/8/g;s/\<C\>/z/g;s/\<E\>/d/g;s/##//g' < $< > $@

clean:
	rm -rf *.pyc *.so
