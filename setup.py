from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("fast_optimization2.metrics", ["fast_optimization2/metrics.pyx"]),
    Extension("fast_optimization2.objective_functions", ["fast_optimization2/objective_functions.pyx"]),
    Extension("fast_optimization2.NSGAII", ["fast_optimization2/NSGAII.pyx"]),
]

setup(
    name='fast_optimization2',
    version='0.1.9',
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='',
    url='https://github.com/defreitasL/fas_optimization2',
    packages=["fast_optimization2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)