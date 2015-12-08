include ../compile_opts/$(ENLIB_COMP).mk
all: pyactgetdata.so tidy
pyactgetdata.so: pyactgetdata.pyx cactgetdata.pxd cactgetdata.h
	$(PYTHON) setup.py build_ext --inplace
clean:
	rm -rf *.so *.pyc
tidy:
	rm -rf build pyactgetdata.c
