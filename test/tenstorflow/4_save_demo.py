#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Saver保存和读取
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# 保存文件
# 注意定义一样的dtype 和数据结构
# W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weight')
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, 'my_net/save_net.ckpt')
#     print('Save to path: ', save_path)


# 读取数据
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weight')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'my_net/save_net.ckpt')
    print('weight:', sess.run(W))
    print('biases:', sess.run(b))