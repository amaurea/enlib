export LAPACK_LINK = -L$(MKLPATH) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread
export OMP_LINK    = -liomp5
export F90FLAGS    = -openmp -O3 -fPIC
export FFLAGS      = $(F90FLAGS)
export FC          = ifort
export F2PY        = f2py
export F2PYCOMP    = intelem

