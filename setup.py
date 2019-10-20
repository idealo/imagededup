import sys
from setuptools import setup, find_packages, Extension

long_description = '''
imagededup is a python package that provides functionality to find duplicates in a collection of images using a variety
of algorithms. Additionally, an evaluation and experimentation framework, is also provided. Following details the
functionality provided by the package:

* Finding duplicates in a directory using one of the following algorithms:
    - Convolutional Neural Network
    - Perceptual hashing
    - Difference hashing
    - Wavelet hashing
    - Average hashing
* Generation of features for images using one of the above stated algorithms.
* Framework to evaluate effectiveness of deduplication given a ground truth mapping.
* Plotting duplicates found for a given image file.

Read the documentation at: https://idealo.github.io/imagededup/

imagededup is compatible with Python 3.6 and is distributed under the Apache 2.0 license.
'''

# Cython compilation is not enabled by default
# http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

# Check whether we're on OSX or not
on_mac = True if sys.platform == 'darwin' else False

ext_modules = []
if use_cython and not on_mac:
    ext_modules += cythonize([
        Extension(
            'brute_force_cython_ext',
            ['imagededup/handlers/search/brute_force_cython_ext.pyx'],
            language='c++',
            extra_compile_args=['-O3', '-march=native', '-mtune=native'],
            extra_link_args=['-O3', '-march=native', '-mtune=native'],
        )
    ])
elif use_cython and on_mac:
    # On Mac, use libc++ because Apple deprecated use of libstdc
    ext_modules += cythonize([
        Extension(
            'brute_force_cython_ext',
            ['imagededup/handlers/search/brute_force_cython_ext.pyx'],
            language='c++',
            extra_compile_args=['-O3', '-march=native', '-mtune=native', '-stdlib=libc++'],
            extra_link_args=['-O3', '-march=native', '-mtune=native', '-lc++', '-nodefaultlibs'],
        )
    ])
else:
    if not on_mac:
        ext_modules += [Extension('brute_force_cython_ext',
                                  ['imagededup/handlers/search/brute_force_cython_ext.cpp'],
                                  )
                        ]
    else:
        ext_modules += [Extension('brute_force_cython_ext',
                                  ['imagededup/handlers/search/brute_force_cython_ext.cpp'],
                                  extra_compile_args=['-stdlib=libc++'],
                                  extra_link_args=['-lc++', '-nodefaultlibs'],
                                  )
                        ]

setup(
    name='imagededup',
    version='0.1.0',
    author='Tanuj Jain, Christopher Lennan, Zubin John, Dat Tran',
    author_email='tanuj.jain.10@gmail.com, christopherlennan@gmail.com, zrjohn@yahoo.com, datitran@gmail.com',
    description='Package for image deduplication',
    long_description=long_description,
    license='Apache 2.0',
    install_requires=[
        'setuptools',
        'numpy<1.17',
        'Pillow<7.0.0',
        'PyWavelets~=1.0.3',
        'scipy',
        'tensorflow~=2.0.0',
        'tqdm',
        'scikit-learn',
        'matplotlib',
    ],
    setup_requires=[
        'cython>=0.29',
    ],
    extras_require={
        'tests': ['pytest', 'pytest-cov', 'pytest-mock', 'codecov'],
        'docs': ['mkdocs', 'mkdocs-material'],
        'dev': ['bumpversion', 'twine', 'cython>=0.29'],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('tests',)),
    ext_modules=ext_modules
)
