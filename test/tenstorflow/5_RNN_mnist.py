#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RNN 循环神经网络
根据语境和之前预测的数据，预测之后的数据含义。
例如: 用语句描述图片含义、 中英文语句翻译、 电脑作曲

LSTM 长短期记忆 cell
主线支线，通过三个门，判断主支线平衡 (writeGate, ForgetGate, ReadGate)

"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 参数定义
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28               # MNIST data input (img shape: 28*28) 28行
n_steps = 28                # time steps 28列
n_hidden_units = 128        # neurons in hidden layer 隐藏层
n_classes = 10              # MNIST classes (0-9 digits) 输出0-9数据

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    # 使用 basic LSTM Cell.  state_is_tuple state是否分为主线支线元祖
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 主线支线元祖(c_state, m_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化记忆全零 state
    # dynamic_rnn 效果不错  time_major 时间维度是否为第一维度，例子是第二维度28 steps  (128 batches, 28 steps, 128 hidden)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)


    # 隐藏层输出数据
    # state 元祖 (c_state, m_state)， final_state[1] 为m_state支线
    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # OR 解开数据把 outputs 变成 列表 [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1


