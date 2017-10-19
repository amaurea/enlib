include ../compile_opts/$(ENLIB_COMP).mk
export CPATH := .:$(CPATH)
export LPATH := .:$(LPATH)

all: iers.py _iers.so libiers.so

iers.py _iers.so: iers.i setup.py iers.c
	swig -python $<
	$(PYTHON) setup.py build_ext --inplace

libiers.so: iers.c
	$(CC) -shared -o $@ -fPIC $<

iers.c: iers2struct finals.all
	./iers2struct < finals.all > $@

iers2struct: iers2struct.c
	$(CC) -o $@ $<

finals.all:
	#curl -k -o finals.all "https://datacenter.iers.org/eop/-/somos/5Rgv/latest/9"
	wget --no-check-certificate -O finals.all "https://datacenter.iers.org/eop/-/somos/5Rgv/latest/9"

clean: tidy
	rm -rf libiers.so _iers.so iers.py *.pyc finals.all
tidy:
	rm -rf iers2struct iers_wrap.c build iers.c
