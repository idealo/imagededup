from setuptools import setup, find_packages

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

setup(
    name='imagededup',
    version='0.0.4',
    author='Tanuj Jain, Christopher Lennan, Zubin John, Dat Tran',
    author_email='tanuj.jain.10@gmail.com, christopherlennan@gmail.com, zrjohn@yahoo.com, datitran@gmail.com',
    description='Package for image deduplication',
    long_description=long_description,
    license='Apache 2.0',
    install_requires=[
        'numpy==1.17.2',
        'Pillow==6.2.0',
        'PyWavelets==1.0.3',
        'scipy==1.3.1',
        'tqdm==4.36.1',
        'scikit-learn==0.21.3',
        'matplotlib==3.1.1',
    ],
    extras_require={
        'tests': ['pytest==5.2.0', 'pytest-cov==2.7.1', 'pytest-mock==1.11.0', 'tensorflow==2.0.0'],
        'docs': ['mkdocs==1.0.4', 'mkdocs-material==4.4.3'],
        'dev': ['bumpversion==0.5.3'],
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
)
