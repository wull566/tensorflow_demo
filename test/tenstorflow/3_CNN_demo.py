#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNN 卷积神经网络

学习准确率达到: 96-97%
AdamOptimizer： 学习速率要低于 0.001  否则容易梯度消失

"""
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 预测准确率
def compute_accuracy(v_xs, v_ys):
    global predition
    y_pre = sess.run(predition, feed_dict={xs:v_xs, keep_prob: [1.]})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob: [1.]})
    return result


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# CNN 调用函数，
def conv2d(x, W):
    # strides步长 4个值 [1, x_movement, y_movement, 1], 0, 1位置必须为1
    # padding 两种配置"SAME" 抽取一样大, "VALID" 抽取略偏小
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# pooling 池化， pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层
def max_pool_2x2(x):
    # tf.nn.max_pool: 最大值池化，tf.nn.avg_pool: 平均值池化
    # strides步长 4个值 [1, x_movement, y_movement, 1], 0, 1位置必须为1
    # padding 两种配置"SAME" 抽取一样大, "VALID" 抽取略偏小
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# 定义占用符

xs = tf.placeholder(tf.float32, [None, 784])    # 28*28
ys = tf.placeholder(tf.float32, [None, 10])    # 10个数字
keep_prob = tf.placeholder(tf.float32)      # dropout 随机忽略隐藏层比率

x_image = tf.reshape(xs, [-1, 28, 28, 1])
print(x_image.shape)  # [n_samples, 28, 28, 1]

# 定义CNN 1
W_conv1 = weight_variable([5, 5, 1, 32])    # 每次抽取patch 5x5图片, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 输出 28x28x32, padding：SAME 导致大小不变
h_pool1 = max_pool_2x2(h_conv1)     # 输出 14x14x32  由于strides=[1,2,2,1] 导致长宽缩小2倍

# 定义CNN 2
W_conv2 = weight_variable([5, 5, 32, 64])    # 每次抽取patch 5x5图片, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 输出 14x14x64, padding：SAME 导致大小不变
h_pool2 = max_pool_2x2(h_conv2)     # 输出 7x7x64  由于strides=[1,2,2,1] 导致长宽缩小2倍

# 定义隐藏层layer 1
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_mb1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(h_mb1)  #矩阵相乘并加上bias偏移量矩阵
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 定义隐藏层layer 2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
predition = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  #矩阵相乘并加上bias偏移量矩阵

# lose
cross_sum = -tf.reduce_sum(ys * tf.log(predition), 1)
cross_entropy = tf.reduce_mean(cross_sum, 0)

train_step = tf.train.AdamOptimizer(0.0004).minimize(cross_entropy)

# 视图
# tf.summary.scalar('loss', cross_entropy)    # 可视化lose 进Event

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # session 初始化
    # tensorboard 摘要可视化启动
    # train_writer = tf.summary.FileWriter("logs/mnist/train", sess.graph)  # 写入可视化日志
    # test_writer = tf.summary.FileWriter("logs/mnist/test", sess.graph)
    # merge = tf.summary.merge_all()      # 自动化处理数据摘要

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 下一批数据
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: [1]})
        # print(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: [1]}))
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
        # if i % 10 == 0:
            # print(sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}))
            # print(compute_accuracy(mnist.test.images, mnist.test.labels))

            # 计算可视化摘要
            # train_res = sess.run(merge, feed_dict={xs: batch_xs, ys: batch_ys})
            # test_res = sess.run(merge, feed_dict={xs: batch_xs, ys: batch_ys})
            # train_writer.add_summary(train_res, i)
            # test_writer.add_summary(test_res, i)


