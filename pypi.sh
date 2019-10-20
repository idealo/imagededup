#!/usr/bin/env bash

rm -rf dist/
pip install -e ".[dev]"
python setup.py sdist bdist_wheel
twine upload dist/*
