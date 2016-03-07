include ../compile_opts/$(ENLIB_COMP).mk
all: sharp.so tidy
sharp.so: sharp.pyx csharp.pxd csharp.h
	$(PYTHON) setup.py build_ext --inplace
clean:
	rm -rf *.so *.pyc
tidy:
	rm -rf build sharp.c
