export LAPACK_LINK = -L$(MKLPATH) -lmkl_rt -lpthread -lm
export OMP_LINK    = -liomp5
export FFLAGS      = -openmp -Ofast -fPIC
export FSAFE       = -openmp -O3 -fPIC
export FC          = ifort
export F2PY        = f2py
export F2PYCOMP    = intelem

