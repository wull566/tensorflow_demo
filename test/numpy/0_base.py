#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numpy 学习

"""
from __future__ import print_function

import numpy as np

array = np.array([[1,2,3],
                  [2,3,4],
                  [2,3,4]], dtype=np.float)

print(array)
print('维度:', array.ndim)
print('行数和列数:', array.shape)
print('个数:', array.size)
print('dtype:', array.dtype)    # 默认 np.int32， np.float64

a = np.arange(1, 100, 5).reshape((-1, 5))
print('a:', a)