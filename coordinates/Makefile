all: pyfsla.so tidy

pyfsla.so: pyfsla.f90 libsla.so
	python setup.py build_ext --inplace

libsla.so:
	$(FC) -O2 -shared -fPIC -o $@ slalib/*.f

clean: tidy
	rm -rf pyfsla.so libsla.so *.pyc
tidy:
	rm -rf build
