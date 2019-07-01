from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
	name="cy_parallax",
	cmdclass = {"build_ext": build_ext},
	ext_modules = [
		Extension(
			name="cy_parallax",
			sources=["cy_parallax.pyx","parallax.c"],
			include_dirs=[".",np.get_include()],
			)
		]
	)
