include ../compile_opts/$(ENLIB_COMP).mk
all:fortran.so
%.so: %.f90
	FFLAGS="$(FSAFE)" $(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $< $(OMP_LINK)
clean:
	rm -rf *.pyc *.so
