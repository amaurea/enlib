include ../compile_opts/$(ENLIB_COMP).mk

all: pyfsla.so tidy

pyfsla.so: pyfsla.f90 libsla.so
	python setup.py build_ext --inplace --fcompiler=$(F2PYCOMP)

libsla.so:
	$(FC) -shared -fPIC -o $@ slalib/*.f

clean: tidy
	rm -rf pyfsla.so libsla.so *.pyc
tidy:
	rm -rf build
