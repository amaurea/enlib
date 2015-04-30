include ../compile_opts/$(ENLIB_COMP).mk
slasrc=$(wildcard slalib/*.f)
slaobj=$(slasrc:.f=.o)

all: pyfsla.so tidy

pyfsla.so: pyfsla.f90 libsla.a
	$(PYTHON) setup.py build_ext --inplace --fcompiler=$(F2PYCOMP)

libsla.a: $(slaobj)
	ar rcs $@ $^

%.o: %.f
	$(FC) $(FFLAGS) -c -o $@ $<

clean: tidy
	rm -rf pyfsla.so libsla.a *.pyc slalib/*.o
tidy:
	rm -rf build
