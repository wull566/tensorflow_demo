#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
过拟合问题
L1,L2...: cost=(Wx - realy)^2 + (W)^2
Dropout: 神经网络过拟合，随机忽略部分隐藏层

"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 添加神经层方法
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weights')   #定义随机变量
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='biases')    # 推荐不为零的数据， 0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases    # tf.matmul 矩阵相乘，把n输入转换为m输出加上偏移量
    # Wx_plus_b = tf.nn.dropout(Wx_plus_b, 0.8)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)    # 激活函数
    # 可视化数据图表， histogram 直方图
    # tf.summary.histogram('Weights', Weights)
    # tf.summary.histogram('biases', biases)
    # tf.summary.histogram('outputs', outputs)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global predition
    y_pre = sess.run(predition, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result


xs = tf.placeholder(tf.float32, [None, 784])    # 28*28
ys = tf.placeholder(tf.float32, [None, 10])    # 10个数字

predition = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# lose
rdsum = -tf.reduce_sum(ys * tf.log(predition), 1)
cross_entropy = tf.reduce_mean(rdsum, 0)

train_step = tf.train.AdamOptimizer(0.004).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 视图
# tf.summary.scalar('loss', cross_entropy)    # 可视化lose 进Event

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # session 初始化

    # tensorboard 摘要可视化启动
    # train_writer = tf.summary.FileWriter("logs/mnist/train", sess.graph)  # 写入可视化日志
    # test_writer = tf.summary.FileWriter("logs/mnist/test", sess.graph)
    # merge = tf.summary.merge_all()      # 自动化处理数据摘要

    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 下一批数据
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            # print(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}))
            print(compute_accuracy(mnist.test.images, mnist.test.labels))

            # 计算可视化摘要
            # train_res = sess.run(merge, feed_dict={xs: batch_xs, ys: batch_ys})
            # test_res = sess.run(merge, feed_dict={xs: batch_xs, ys: batch_ys})
            # train_writer.add_summary(train_res, i)
            # test_writer.add_summary(test_res, i)


