pyfsla.so: pyfsla.f90 tidy
	python setup.py build_ext --inplace

clean: tidy
	rm -rf pyfsla.so *.pyc
tidy:
	rm -rf build
