from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	name="pyactgetdata",
	cmdclass = {"build_ext": build_ext},
	ext_modules = [
		Extension(
			name="pyactgetdata",
			sources=["pyactgetdata.pyx","read_parallel.c"],
			libraries=["actgetdata","slim","zzip"],
			include_dirs=["."],
			)
		]
	)
