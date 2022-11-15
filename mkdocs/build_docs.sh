#!/usr/bin/env bash

cp ../README.md docs/index.md
cp ../CONTRIBUTING.md docs/CONTRIBUTING.md
cp ../LICENSE docs/LICENSE.md
cp -R ../readme_figures docs/
echo "Running autogen .."
python autogen.py
echo "Finished autogen .."
# mkdir ../docs
mkdocs build -c -d ../docs/