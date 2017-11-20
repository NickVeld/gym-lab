from distutils.core import setup, Extension

import numpy
from Cython.Build import cythonize

extensions = [Extension("pathenv.utils_compiled",
                        ["pathenv/utils_compiled.pyx"])]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
