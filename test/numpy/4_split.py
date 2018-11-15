#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numpy 学习

"""
from __future__ import print_function

import numpy as np

A = np.arange(12).reshape((3, 4))
"""
array([[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]])
"""

# 均等分割
print(np.split(A, 2, axis=1))
print(np.split(A, 3, axis=0))

# 不均等分割
print(np.array_split(A, 3, axis=1))
print(np.array_split(A, 2, axis=0))

print(np.vsplit(A, 3))
print(np.hsplit(A, 2))

