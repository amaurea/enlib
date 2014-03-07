from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sourcefiles = ['wrapper.pyx', 'read_table.c']
extensions = [Extension("wrapper", sourcefiles)]
setup(ext_modules=cythonize(extensions))

