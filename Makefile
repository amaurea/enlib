all: sharp iers coordinates array_ops pmat nmat
clean: clean_sharp clean_iers clean_coordinates clean_array_ops clean_pmat clean_nmat
	rm -rf *.pyc

sharp: foo
	make -C sharp
clean_sharp:
	make -C sharp clean
iers: foo
	make -C iers
clean_iers:
	make -C iers clean
coordinates: foo
	make -C coordinates
clean_coordinates:
	make -C coordinates clean
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


foo:
