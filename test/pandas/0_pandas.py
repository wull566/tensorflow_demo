#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pandas 学习

字典形式的numpy

"""
from __future__ import print_function

import numpy as np
import pandas as pd

s = pd.Series([1,3,6,np.nan,44,1])
print(s)
'''
0     1.0
1     3.0
2     6.0
3     NaN
4    44.0
5     1.0
'''

dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4),index=dates,columns=['a','b','c','d'])
print(df)

'''
                   a         b         c         d
2016-01-01 -1.551779 -1.669971 -2.021800 -2.433610
2016-01-02  0.177421 -1.377563  2.674057 -0.463060
2016-01-03  1.908429  1.463810  2.691219  0.328285
2016-01-04  1.965028 -0.834849 -1.637209  0.789990
2016-01-05  0.185888 -0.169179  1.100780  1.686840
2016-01-06  0.173937 -0.448993  2.002772 -0.940095
'''

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

print(df2)

print(df2.index)
print(df2.columns)
print(df2.dtypes)
print(df2.values)
print(df2.describe())   # 直接运算出数值列的所有概要

print(df2.T)
print(df2.sort_index(axis=1, ascending=False))      # ascending = False 倒序
print(df2.sort_index(axis=0, ascending=False))
print(df2.sort_values(by='E'))

