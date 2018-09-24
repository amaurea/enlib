include ../compile_opts/$(ENLIB_COMP).mk
all: sharp.so tidy
sharp.so: sharp.c csharp.h
#Alex added this statement -- if you have compiled sharp with icc it seems you need to have this
ifeq ($(CC),icc)
	LDSHARED="icc -shared" CC=icc $(PYTHON) setup.py build_ext --inplace
else
        $(CC) -o foo $(objects) $(normal_libs)
endif


# $(PYTHON) setup.py build_ext --inplace

sharp.c: sharp.pyx csharp.pxd
	cython sharp.pyx
	perl -pi -e 's/typedef npy_float64 _Complex/typedef double _Complex/;s/typedef npy_float32 _Complex/typedef float _Complex/' sharp.c
clean:
	rm -rf *.so *.pyc sharp.c
tidy:
	rm -rf build
