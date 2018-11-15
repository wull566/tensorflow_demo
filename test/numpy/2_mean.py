#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numpy 学习

"""
from __future__ import print_function

import numpy as np

a = np.arange(2, 14).reshape((3,4))

# a:[[ 2  3  4  5]
#  [ 6  7  8  9]
#  [10 11 12 13]]

print('a:', a)
print('np.argmin(a):', np.argmin(a))    # 最小元素索引
print('np.argmax(a):', np.argmax(a))    # 最大元素索引

print('np.mean(a):', np.mean(a))    # 平均值 7.5
print('np.median(a):', np.median(a))    # 中位数 7.5
print('np.cumsum(a):', np.cumsum(a))    # 累加函数  [2 5 9 14 20 27 35 44 54 65 77 90]
print('np.diff(a):', np.diff(a))    # 累差 [[1 1 1] [1 1 1] [1 1 1]]
print('np.nonzero(a):', np.nonzero(a))    # 输出非零的行列值: (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
b = np.arange(14, 2, -1).reshape((3,4))
print('np.sort(b):', np.sort(b))    # 正排序
print('np.transpose(a):', np.transpose(a))    # 矩阵的转置 a.T
print('np.clip(a):', np.clip(a, 3, 9))    # 缩减矩阵值区间大小，[min, max] 保持区间内

# 都有矩阵 参数 aixs: 0, 列计算， 1，行计算

