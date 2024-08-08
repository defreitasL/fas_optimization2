from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([
        "fast_optimization2/metrics.pyx",
        "fast_optimization2/objective_functions.pyx",
        "fast_optimization2/NSGAII.pyx"
    ]),
    include_dirs=[numpy.get_include()]
)


