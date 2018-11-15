#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pandas 学习

字典形式的numpy

"""
from __future__ import print_function

import numpy as np
import pandas as pd

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])

"""
             A   B   C   D
2013-01-01   0   1   2   3
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""

df.to_csv('datas.csv')
rdf = pd.read_csv('datas.csv')
print(rdf)

#读取csv
std = pd.read_csv('student.csv')

#打印出data
print(std)

# std.to_pickle('student.pickle')
