#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MongoDB 练习

"""

from __future__ import print_function
from pymongo import MongoClient

# 192.168.0.145:27017
conn = MongoClient('192.168.0.145', 27017)
db = conn.py_test
myuser = db.myuser

myuser.save({"name":"zhangsan","age":18})

for user in myuser.find():
    print('myuser', user)
