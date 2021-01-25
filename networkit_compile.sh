#!/bin/bash

cd networkit

python setup.py build_ext
pip install -e .