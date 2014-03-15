from numpy.distutils.core import setup, Extension
wrapper = Extension('pyfsla', sources=['pyfsla.f90'], libraries=['sla'])
setup(ext_modules = [wrapper])
