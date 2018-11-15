"""
强化学习
案例： Alpha go围棋，自己学习玩经典游戏

强化学习方法:
不理解环境(Model-Free RL)
理解环境(Model-Based RL), 多个想象力，能想象各种反馈预判，再执行概率比较好的执行，等待反馈

Q Learning, Sarsa, Policy Gradients,

基于概率：
选择各种行为的概率可能性，概率随机选择动作
Policy Gradients

基于价值：
选择价值最高的行为
Q Learning, Sarsa

合并两者:
Actor-Critic 基于Actor概率选择动作，Critic根据行为算出价值

类型:
回合更新： Policy Gradients(基础版)， Monte-Carlo Learning
单步更新(更优): Policy Gradients(升级), Q Learning, Sarsa

在线学习: Sarsa
离线学习: Q Learning, Deep Q Network

强化学习知识要求
Numpy，Pandas 必学， 用于数据处理
Matplotlib 可选， 用于绘图
Tkinter 可选， 用户强化学习的模拟环境编写
tenstorflow, 神经网络和强化网络结合
OpenApi gym 可选， 提供现成的模拟环境，不支持windows


"""
import gym
import numpy as np
import tensorflow as tf

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # 1_tensorflow_new rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
MEMORY_COUNTER = 0          # for store experience
LEARNING_STEP_COUNTER = 0   # for target updating
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
MEMORY = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory

# tf placeholders
tf_s = tf.placeholder(tf.float32, [None, N_STATES])
tf_a = tf.placeholder(tf.int32, [None, ])
tf_r = tf.placeholder(tf.float32, [None, ])
tf_s_ = tf.placeholder(tf.float32, [None, N_STATES])

with tf.variable_scope('q'):        # evaluation network
    l_eval = tf.layers.dense(tf_s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
    q = tf.layers.dense(l_eval, N_ACTIONS, kernel_initializer=tf.random_normal_initializer(0, 0.1))

with tf.variable_scope('q_next'):   # target network, not to train
    l_target = tf.layers.dense(tf_s_, 10, tf.nn.relu, trainable=False)
    q_next = tf.layers.dense(l_target, N_ACTIONS, trainable=False)

q_target = tf_r + GAMMA * tf.reduce_max(q_next, axis=1)                   # shape=(None, ),

a_indices = tf.stack([tf.range(tf.shape(tf_a)[0], dtype=tf.int32), tf_a], axis=1)
q_wrt_a = tf.gather_nd(params=q, indices=a_indices)     # shape=(None, ), q for current state

loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def choose_action(s):
    s = s[np.newaxis, :]
    if np.random.uniform() < EPSILON:
        # forward feed the observation and get q value for every actions
        actions_value = sess.run(q, feed_dict={tf_s: s})
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, N_ACTIONS)
    return action


def store_transition(s, a, r, s_):
    global MEMORY_COUNTER
    transition = np.hstack((s, [a, r], s_))
    # replace the old memory with new memory
    index = MEMORY_COUNTER % MEMORY_CAPACITY
    MEMORY[index, :] = transition
    MEMORY_COUNTER += 1


def learn():
    # update target net
    global LEARNING_STEP_COUNTER
    if LEARNING_STEP_COUNTER % TARGET_REPLACE_ITER == 0:
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
        sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
    LEARNING_STEP_COUNTER += 1

    # 1_tensorflow_new
    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
    b_memory = MEMORY[sample_index, :]
    b_s = b_memory[:, :N_STATES]
    b_a = b_memory[:, N_STATES].astype(int)
    b_r = b_memory[:, N_STATES+1]
    b_s_ = b_memory[:, -N_STATES:]
    sess.run(train_op, {tf_s: b_s, tf_a: b_a, tf_r: b_r, tf_s_: b_s_})

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        store_transition(s, a, r, s_)

        ep_r += r
        if MEMORY_COUNTER > MEMORY_CAPACITY:
            learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_