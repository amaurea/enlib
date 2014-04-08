from distutils.core import setup
from distutils.extension import Extension

setup(name = "iers",
		ext_modules=[Extension("_iers", sources=["iers.c","iers_wrap.c"])],
		py_modules=["iers"])
