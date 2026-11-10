"""
Print the arrays stored in prediction files (read only).

Usage: python npz_keys.py <file> [<file> ...]
"""

import sys

import numpy

for path in sys.argv[1:]:
    with numpy.load(path) as content:
        print(path)
        for key in content.files:
            array = content[key]
            print(f"  {key}: shape {array.shape}, dtype {array.dtype}, first {array.ravel()[:3]}")
