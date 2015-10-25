from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	name="actgetdata",
	cmdclass = {"build_ext": build_ext},
	ext_modules = [
		Extension(
			name="actgetdata",
			sources=["actgetdata.pyx"],
			libraries=["actgetdata","slim","zzip"],
			include_dirs=["."],
			)
		]
	)
