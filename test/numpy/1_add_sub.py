#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
numpy 学习

"""
from __future__ import print_function

import numpy as np

a = np.array([[10,20,30, 40]])
b = np.arange(4)

print('1@ ', 'a * b = ', a, b, a * b)


a = np.array([[1,1],
              [0,1]])
b = np.array([[1,2],
              [3,4]])

print('相乘: ', 'a * b = ', a * b)        #[[1 2], [0 4]]
print('矩阵相乘: ', 'np.dot() ', np.dot(a, b))  #[[4 6], [3 4]]