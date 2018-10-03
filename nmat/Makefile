include ../compile_opts/$(ENLIB_COMP).mk

all: nmat_core_32.so nmat_core_64.so

%.so: %.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $< $(LAPACK_LINK) $(OMP_LINK)

nmat_core_32.f90: nmat_core.F90
	perl -pe 's/\b_\b/4/g;s/\bC\b/c/g;s/\bD\b/e/g;s/S/s/g;s/##//g' < $< > $@
nmat_core_64.f90: nmat_core.F90
	perl -pe 's/\b_\b/8/g;s/\bC\b/z/g;s/\bD\b/d/g;s/S/d/g;s/##//g' < $< > $@

clean:
	rm -rf *.pyc *.so
