include ../compile_opts/$(ENLIB_COMP).mk
all:
	$(PYTHON) setup.py build_ext --inplace $(OMP_LINK)
clean:
	rm -rf *.pyc *.so build __pycache__ cy_parallax.c
