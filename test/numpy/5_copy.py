#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numpy 学习

"""
from __future__ import print_function

import numpy as np

a = np.arange(4)
"""
array([ 0,  1,  2])
"""

# b = a
b = a.copy()
c = a
d = b
a[0] = 11

print(a, b, c, d)


