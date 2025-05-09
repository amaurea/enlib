export LAPACK_LINK =  -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm
export OMP_LINK    = -lgomp
export FFLAGS      = -fopenmp -Ofast -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs -fallow-argument-mismatch
#export FFLAGS      = -fopenmp -O0 -g -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs -fallow-argument-mismatch -fbounds-check
export FSAFE       = -fopenmp -O3 -fPIC -ffree-line-length-none -fdiagnostics-color=always -Wno-tabs -fallow-argument-mismatch 
export CFLAGS      = -fopenmp -lgomp -fPIC -std=c99 -march=native
export FC          = gfortran
export F2PY        = f2py
export F2PYCOMP    = gnu95
export PYTHON      = python
export SED         = sed
export CC          = gcc
export LDSHARED    = $(CC) -shared
