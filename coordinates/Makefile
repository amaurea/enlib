include ../compile_opts/$(ENLIB_COMP).mk
slasrc=$(wildcard slalib/*.f)

all: pyfsla.so tidy

pyfsla.so: pyfsla.f90 $(slasrc)
	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $^ $(OMP_LINK)

clean: tidy
	rm -rf pyfsla.so libsla.a *.pyc slalib/*.o
tidy:
	rm -rf build
