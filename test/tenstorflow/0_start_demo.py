#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器学习概论

监督学习
非监督学习
半监督学习
强化学习：投篮命中强化
遗传算法： 通过淘汰弱者进行强化

人工神经网络:

"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data
# 创建100个随机数列， 0-1的范围
x_data = np.random.rand(100).astype(np.float32)
print(x_data)

# 需要被学习成为的函数结构
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
# 生成Weights初始值为 -1 到 1 的随机1维度矩阵
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 生成biases初始值为 0 的 1维度矩阵
biases = tf.Variable(tf.zeros([1]))

# 预测y的值
y = Weights*x_data + biases

# 预测y 和 y_dada 差别，即损失值
loss = tf.reduce_mean(tf.square(y-y_data))

#选择基本Optimizer学习模块，效率为0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
### create tensorflow structure end ###

# 开始初始化环境，激活网络
sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# 设置执行训练次数，并运行打印结果
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases), sess.run(loss))