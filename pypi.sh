#!/usr/bin/env bash

rm -rf dist/
pip install ".[dev]"
python setup.py sdist bdist_wheel
twine upload dist/*
