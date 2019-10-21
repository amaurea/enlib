all: iers coordinates sampcut array_ops pmat nmat ptsrc_data pyactgetdata cython
clean: clean_iers clean_coordinates clean_sampcut clean_array_ops clean_pmat clean_nmat clean_ptsrc_data clean_pyactgetdata clean_cython
	rm -rf *.pyc

iers: foo
	make -C iers
clean_iers:
	make -C iers clean
coordinates: foo
	make -C coordinates
clean_coordinates:
	make -C coordinates clean
sampcut: foo
	make -C sampcut
clean_sampcut:
	make -C sampcut clean
array_ops: foo
	make -C array_ops
clean_array_ops:
	make -C array_ops clean
pmat: foo
	make -C pmat
clean_pmat:
	make -C pmat clean
nmat: foo
	make -C nmat
clean_nmat:
	make -C nmat clean
ptsrc_data: foo
	make -C ptsrc_data
clean_ptsrc_data:
	make -C ptsrc_data clean
pyactgetdata: foo
	make -C pyactgetdata
clean_pyactgetdata:
	make -C pyactgetdata clean
cython: foo
	make -C cython
clean_cython:
	make -C cython clean

foo:
