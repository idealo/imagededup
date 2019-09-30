from setuptools import setup, find_packages

long_description = '''
imagededup is a python  package that provides functionality to find duplicates in a collection of images using a variety
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
'''

setup(
    name='imagededup',
    version='0.0.2',
    packages=find_packages(exclude=('tests',)),
    url='',
    long_description=long_description,
    license='Apache 2.0',
    author='Tanuj Jain, Christopher Lennan, Zubin John',
    author_email='tanuj.jain.10@gmail.com, christopherlennan@gmail.com, zrjohn@yahoo.com',
    description='Package for image deduplication',
    install_requires=[
        'numpy==1.16.3',
        'Pillow==6.0.0',
        'PyWavelets==1.0.3',
        'scipy==1.2.1',
        'keras==2.2.4',
        'tensorflow==1.13.1',
        'tqdm==4.35.0',
        'scikit-learn==0.21.2',
        'matplotlib==3.1.1',
    ],
    extras_require={
        'tests': ['pytest==4.4.1', 'pytest-cov==2.6.1', 'pytest-mock==1.10.4'],
        'docs': ['mkdocs==1.0.4', 'mkdocs-material==4.0.2'],
        'dev': ['bumpversion==0.5.3'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
