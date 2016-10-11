from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	name="sharp",
	cmdclass = {"build_ext": build_ext},
	ext_modules = [
		Extension(
			name="sharp",
			sources=["sharp.c"],
			libraries=["sharp","c_utils","fftpack"],
			include_dirs=["."],
			extra_link_args = ["-fopenmp"],
			)
		]
	)
