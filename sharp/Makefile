sharp.so: sharp.pyx csharp.pxd csharp.h tidy
	python setup.py build_ext --inplace
clean:
	rm -rf *.so *.pyc
tidy:
	rm -rf build sharp.c
