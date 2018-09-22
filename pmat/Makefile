include ../compile_opts/$(ENLIB_COMP).mk
skip := skip: map_block_prepare map_block_finish map_block_prepare_shifted_flat map_block_finish_shifted_flat map_block_prepare_direct_flat map_block_finish_direct_flat :
all: pmat_core_32.so pmat_core_64.so

%.so: %.f90
	$(F2PY) --fcompiler=$(F2PYCOMP) --noopt -c -m $(basename $<) $< $(skip) $(OMP_LINK)
	#FFLAGS="-fPIC -O0 -g -fbounds-check -Wno-tabs -ffree-line-length-none" $(F2PY) --noopt --fcompiler=gfortran -c -m $(basename $<) $< $(skip) -lgomp

pmat_core_32.f90: pmat_core.F90
	perl -pe 's/\b_\b/4/g' < $< > $@
pmat_core_64.f90: pmat_core.F90
	perl -pe 's/\b_\b/8/g' < $< > $@

clean:
	rm -rf *.pyc *.so
