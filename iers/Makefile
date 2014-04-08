export CPATH := .:$(CPATH)
export LPATH := .:$(LPATH)

all: iers.py _iers.so libiers.so tidy

iers.py _iers.so: iers.i setup.py iers.c
	swig -python $<
	python setup.py build_ext --inplace

libiers.so: iers.c
	gcc -shared -o $@ -fPIC $<

iers.c: iers2struct finals.all
	./iers2struct < finals.all > $@

iers2struct: iers2struct.c
	gcc -o $@ $<

finals.all:
	curl -o finals.all http://datacenter.iers.org/eop/-/somos/5Rgv/latest/9

clean: tidy
	rm -rf libiers.so _iers.so iers.py
tidy:
	rm -rf finals.all iers2struct iers_wrap.c build iers.c
