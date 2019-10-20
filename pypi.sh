#!/usr/bin/env bash

rm -rf dist/
pip install -r requirements.txt
python setup.py sdist bdist_wheel
twine upload dist/*
