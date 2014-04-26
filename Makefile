all: sharp/sharp.so iers/_iers.so coordinates/pyfsla.so
clean: clean_sharp clean_iers clean_coordinates
	rm -rf *.pyc

sharp/sharp.so: foo
	make -C sharp
clean_sharp:
	make -C sharp clean
iers/_iers.so: foo
	make -C iers
clean_iers:
	make -C iers clean
coordinates/pyfsla.so: foo
	make -C coordinates
clean_coordinates:
	make -C coordinates clean

foo:
