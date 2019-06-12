from setuptools import setup

setup(
    name='image-dedup',
    version='',
    packages=['imagededup'],
    url='',
    license='MIT',
    author='Tanuj Jain, Zubin John',
    author_email='tanuj.jain.10@gmail.com, zrjohn@yahoo.com',
    description='Package for image deduplication',
    install_requires=['numpy==1.16.3', 'Pillow==6.0.0', 'PyWavelets==1.0.3', 'scipy==1.2.1'],
    extras_require={
        'tests': ['pytest==4.4.1', 'pytest-cov==2.6.1', 'pytest-mock==1.10.4']
    }
)
