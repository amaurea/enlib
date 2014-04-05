sharp.so: sharp.pyx csharp.pxd csharp.h
	python setup.py build_ext --inplace && rm -rf build # sharp.c
clean:
	rm -rf build sharp.c *.so *.pyc
