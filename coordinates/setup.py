from numpy.distutils.core import setup, Extension
wrapper = Extension('pyfsla', sources=['pyfsla.f90'], extra_objects=['libsla.a'])
setup(ext_modules = [wrapper])
