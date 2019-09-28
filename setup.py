from setuptools import setup

setup(
    name='image-dedup',
    version='',
    packages=['imagededup'],
    url='',
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
    packages=find_packages(exclude=('tests',))
)
