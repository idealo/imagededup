#!/usr/bin/env bash

rm -rf dist/

pip install "cython>=0.29"
python setup.py sdist bdist_wheel
twine upload dist/*
