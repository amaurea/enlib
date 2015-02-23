export LAPACK_LINK = -llapack -lopenblas
export OMP_LINK    = -lgomp
export FFLAGS      = -fopenmp -O3 -ffast-math -fPIC -ffree-line-length-none
export FSAFE       = -fopenmp -O3 -fPIC -ffree-line-length-none
export FC          = gfortran
export F2PY        = f2py
export F2PYCOMP    = gfortran
