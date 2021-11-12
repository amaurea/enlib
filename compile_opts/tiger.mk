export LAPACK_LINK = -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm
export OMP_LINK    = -lgomp
export FFLAGS      = -qopenmp -Ofast -fPIC -xhost -nofor-main # -vec-report -opt-report
#export FFLAGS      = -O0 -traceback -check bounds -g -fPIC -xhost -nofor-main # -vec-report -opt-report
export FSAFE       = -qopenmp -O3 -fPIC -xhost
export CFLAGS      = -fopenmp -lgomp -fPIC -std=c99
export FC          = ifort
export F2PY        = f2py
export F2PYCOMP    = intelem
export PYTHON      = python
export SED         = sed
export CC          = icc
export LDSHARED    = $(CC) -shared
