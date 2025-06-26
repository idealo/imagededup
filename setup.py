"""import sys
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
COMPILE_LINK_ARGS = ['-O3', '-march=native', '-mtune=native']
COMPILE_ARGS_OSX = ['-stdlib=libc++']
LINK_ARGS_OSX = ['-lc++', '-nodefaultlibs']

ext_modules = []
source_ext = '.pyx' if use_cython else '.cpp'
source_file = MOD_PATH + source_ext

compile_args = COMPILE_LINK_ARGS.copy()
link_args = COMPILE_LINK_ARGS.copy()

if on_mac:
    compile_args += COMPILE_ARGS_OSX
    link_args += LINK_ARGS_OSX

ext = Extension(
    MOD_NAME,
    [source_file],
    language='c++',
    extra_compile_args=compile_args,
    extra_link_args=link_args
)

ext_modules = cythonize([ext]) if use_cython else [ext]

setup(ext_modules=ext_modules)
"""
import sys
from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

on_mac = sys.platform.startswith('darwin')
on_windows = sys.platform.startswith('win')

MOD_NAME = 'brute_force_cython_ext'
MOD_PATH = 'imagededup/handlers/search/brute_force_cython_ext'
COMPILE_LINK_ARGS = ['-O3', '-march=native', '-mtune=native']
# On Mac, use libc++ because Apple deprecated use of libstdc
COMPILE_ARGS_OSX = ['-stdlib=libc++']
LINK_ARGS_OSX = ['-lc++', '-nodefaultlibs']

ext_modules = []
if use_cython and on_mac:
    ext_modules += cythonize([
        Extension(
            MOD_NAME,
            [MOD_PATH + '.pyx'],
            language='c++',
            extra_compile_args=COMPILE_LINK_ARGS + COMPILE_ARGS_OSX,
            extra_link_args=COMPILE_LINK_ARGS + LINK_ARGS_OSX,
        )
    ])
elif use_cython and on_windows:
    ext_modules += cythonize([
        Extension(
            MOD_NAME,
            [MOD_PATH + '.pyx'],
            language='c++',
        )
    ])
elif use_cython:
    ext_modules += cythonize([
        Extension(
            MOD_NAME,
            [MOD_PATH + '.pyx'],
            language='c++',
            extra_compile_args=COMPILE_LINK_ARGS,
            extra_link_args=COMPILE_LINK_ARGS,
        )
    ])
else:
    if on_mac:
        ext_modules += [Extension(MOD_NAME,
                                  [MOD_PATH + '.cpp'],
                                  extra_compile_args=COMPILE_ARGS_OSX,
                                  extra_link_args=LINK_ARGS_OSX,
                                  )
                        ]
    else:
        ext_modules += [Extension(MOD_NAME,
                                  [MOD_PATH + '.cpp'],
                                  )
                        ]
