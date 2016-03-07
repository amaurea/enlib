export LAPACK_LINK = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_avx -lmkl_def -lmkl_core -lmkl_intel_thread -lpthread -lm
export OMP_LINK    = -liomp5
export F90FLAGS    = -openmp -Ofast -fPIC
#export F90FLAGS    = -O0 -g -debug all -check bounds -fPIC
export FFLAGS      = $(F90FLAGS)
export FC          = ifort
export F2PY        = f2py
export F2PYCOMP    = intelem

