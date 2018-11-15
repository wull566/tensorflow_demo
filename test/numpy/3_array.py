#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numpy 学习

"""
from __future__ import print_function

import numpy as np

# a = np.arange(3, 15).reshape((3,4))
#
# print('a[1,1:3]', a[1,1:3])
#
# print('a.flatten()', a.flatten())
# for item in a.flat:
#     print(item)


b = np.array([[1,1,1],[2,2,2]])
c = np.array([[3,3,3],[4,4,4]])
d = np.vstack((b,c))      # 上下合并
print('np.vstack((b,c))', d)
print(b.shape, d.shape)

e = np.hstack((b,c))        # 左右合并
print('np.hstack((b,c))', e)
print(b.shape, e.shape)


aa = np.array([1,1,1])[:, np.newaxis]
bb = np.array([3,3,3])[:, np.newaxis]
print('np.hstack((aa,bb))', np.hstack((aa,bb)))

print('np.concatenate((aa,bb))', np.concatenate((b, c), axis=1))