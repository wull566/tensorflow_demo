#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
激励函数
线性方程 y = Wx 和 非线性方程 y = AF(Wx)
激励函数使得线性方程 变成 非线性方程
注意: 特别多层的神经网络不能随意选择，会涉及梯度爆炸和梯度消失的问题

常用选择:
卷积神经网络推荐，relu
循环神经网络， tanh 或 relu

激励函数在隐藏层输出时过激励函数变化

activation

注意： 绘制图片时，直接在PyCharm中运行无动态效果
取消选择 Setting-Tools-Python Scientitic 中的 Show plots in toolwindow

训练优化
SGD：分批分量训练
Momentum: 增加下坡惯性 W = b1 * W - Learning rate * dx   原:  W = - Learning rate * dx
AdaGrad: 增加错误方向阻力
RMSProp: Momentum + AdaGrad 缺少部分参数
Adam: 完美结合 Momentum + AdaGrad

tensorboard 可视化摘要summary:
    tensorboard --logdir D:\python_work\tensorflow_models\test\demo\logs

"""
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 添加神经层方法
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weights')   #定义随机变量
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biases')    # 推荐不为零的数据， 0.1
        Wx_plus_b = tf.matmul(inputs, Weights) + biases    # tf.matmul 矩阵相乘，把n输入转换为m输出加上偏移量

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)    # 激活函数

        # 可视化数据图表， histogram 直方图
        tf.summary.histogram('Weights', Weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('outputs', outputs)
        return outputs


# 返回均匀间隔的数字， 并且添加新维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)    # 噪音
y_data = np.square(x_data) - 0.5 + noise    # x 平方 + 位移 + 偏移噪音

# 占位符，数据分量输出
with tf.name_scope('inputs'):    # 增加可视化模块
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 1个输入数据，10个输出隐藏值, relu激励方程
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 预测值 10个输入数据，1个输出隐藏值
predition = add_layer(l1, 10, 1, activation_function=None)

# 损失函数
# tf.square: 平方
# tf.reduce_sum： 求和, reduction_indices 表示函数的处理维度, 0:一个值，1: 一维列表
# tf.reduce_mean: 平均值
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)    # 可视化lose 进Event

# 训练学习提升
# GradientDescentOptimizer 学习效率0.1步阶
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # session 初始化

    # tensorboard 摘要可视化启动
    writer = tf.summary.FileWriter("logs/linear", sess.graph)  # 写入可视化日志
    merge = tf.summary.merge_all()      # 自动化处理数据摘要

    # 图表显示
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)
    plt.ion()

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # 返回损失值
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

            # 计算可视化摘要
            merged = sess.run(merge, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(merged, i)

            #绘制预测值的线
            try:
                ax.lines.remove(lines[0])  # 移除旧的线
            except Exception:
                pass
            predition_value = sess.run(predition, feed_dict={xs: x_data})
            # print(predition_value)
            lines = ax.plot(x_data, predition_value, 'r-', lw=5)
            plt.pause(0.1)  #暂停0.1秒

    plt.pause(5)
