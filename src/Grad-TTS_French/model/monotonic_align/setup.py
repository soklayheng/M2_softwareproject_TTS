""" from https://github.com/jaywalnut310/glow-tts """
from setuptools import find_packages, setup
from Cython.Build import cythonize
from distutils.core import setup
import numpy


setup(
    name = 'monotonic_align',
    ext_modules = cythonize("core.pyx"),
    script_args=['build'],
    options={'build':{'build_lib':'.'}},
    include_dirs=[numpy.get_include()]
)
