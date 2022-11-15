#!/usr/bin/env bash

cp ../README.md docs/index.md
cp ../CONTRIBUTING.md docs/CONTRIBUTING.md
cp ../LICENSE docs/LICENSE.md
cp -R ../readme_figures docs/
python autogen.py
mkdocs build -c -d ../docs/