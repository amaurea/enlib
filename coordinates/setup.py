from numpy.distutils.core import setup, Extension
wrapper = Extension('pyfsla', sources=['pyfsla.f90'], libraries=['sla'], library_dirs=['.'])
setup(ext_modules = [wrapper])
