import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

on_mac = sys.platform.startswith('darwin')
on_windows = sys.platform.startswith('win')

MOD_NAME = 'brute_force_cython_ext'
MOD_PATH = 'imagededup/handlers/search/brute_force_cython_ext'
COMPILE_ARGS_BASE = ['-O3']
COMPILE_ARGS_OSX = ['-stdlib=libc++']
LINK_ARGS_OSX = ['-lc++', '-nodefaultlibs']
COMPILE_ARGS_NATIVE = ['-march=native', '-mtune=native']

# Start with base args
compile_args = COMPILE_ARGS_BASE.copy()
link_args = COMPILE_ARGS_BASE.copy()

# Only use -march=native if not doing macOS universal builds
if not on_mac and not on_windows:
    compile_args += COMPILE_ARGS_NATIVE
    link_args += COMPILE_ARGS_NATIVE

# macOS-specific adjustments
if on_mac:
    compile_args += COMPILE_ARGS_OSX
    link_args += LINK_ARGS_OSX

ext_modules = []
source_ext = '.pyx' if use_cython else '.cpp'
source_file = MOD_PATH + source_ext

ext = Extension(
    MOD_NAME,
    [source_file],
    language='c++',
    extra_compile_args=compile_args,
    extra_link_args=link_args
)

ext_modules = cythonize([ext]) if use_cython else [ext]

setup(ext_modules=ext_modules)
