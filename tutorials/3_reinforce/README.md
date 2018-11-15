# 强化学习

#### 案例：
 
 Alpha go围棋，自己学习玩经典游戏

#### 强化学习方法:

    不理解环境(Model-Free RL)
    理解环境(Model-Based RL), 多个想象力，能想象各种反馈预判，再执行概率比较好的执行，等待反馈

Q Learning, Sarsa, Policy Gradients,

#### 强化学习行为选择

* 基于概率：
    选择各种行为的概率可能性，概率随机选择动作
    Policy Gradients

* 基于价值
    选择价值最高的行为
    Q Learning, Sarsa

* 合并两者:
    Actor-Critic 基于Actor概率选择动作，Critic根据行为算出价值

#### 更新类型

    回合更新： Policy Gradients(基础版)， Monte-Carlo Learning
    单步更新(更优): Policy Gradients(升级), Q Learning, Sarsa

#### 学习方式

    在线学习: Sarsa
    离线学习: Q Learning, Deep Q Network

## 强化学习知识要求

    Numpy，Pandas 必学， 用于数据处理
    Matplotlib 可选， 用于绘图
    Tkinter 可选， 用户强化学习的模拟环境编写
    tenstorflow, 神经网络和强化网络结合
    OpenApi gym 可选， 提供现成的模拟环境，不支持windows
    
    
# 强化学习

## 传统强化学习

#### Q Learning 和 Sarsa 对比：
 
* Q Learning 在行为收益计算时只关心行为最大收益，即达到目的的收益，不考虑其他行为的风险

* Sarsa 计算收益时，会去反馈下一行为概率的所有可能的风险和收益

* Sarsa(lambda) 回合衰变更新方法， lambda 衰变强度 0 - 1 

## DQN : Deep Q Network  神经网络强化学习

    神经网络 + Q learning
    
    DQN 拥有 两个神经网络，eval为一直增长的主线估计的神经网络，target为一定时间固话后的现实神经网络。
    Memory：学习经验数据将会以固定长度的形势存放在内存中，
    格式：(s, [a, r], s_) (行为前状态， 行为， 奖励， 行为后状态)
    leaning学习时，将随机批量抽取Memory记忆中的数据，放入到两个神经网络，
    s_位置的现实网络max最大Q值max_target4next 对 s位置该行为的估计网络数据进行增益,
    增益值: q_eval[s, action] = reward + self.gamma * np.max(q_next, axis=1)
    误差值loss = 修改后的eval估计神经网络 与 target现实神经网络 之间的Q值差异

#### Double DQN 两次估计优化 - 神经网络强化学习

    优化增益值计算：
    
    修改为 先获取eval估计神经网络对于s_位置的Q值估计q_eval4next，
    使用q_eval4next估计s_位置max最大值 代替 max_target4next 对 s位置该行为的估计网络数据进行增益,
    增益值: q_eval[s, action] = reward + self.gamma * np.argmax(q_eval4next, axis=1) 

#### DQN with Prioritized Replay 优先级反馈优化 - 神经网络强化学习

    abs_errors： |Q现实 - Q估计| 的绝对值
    如果 abs_errors 越大, 代表预测精度还有很多上升空间, 就越需要被学习, 优先级 p 越高.
    leaning学习时， 使用优先级P 较高的批量数据 代替 原有的随机批量数据进行计算。
    
    
#### Dueling DQN  状态行为拆分优化 - 神经网络强化学习

    将每个动作的 Q 拆分成了 state 收益 加上 每个动作的 收益.
    
    # V 为状态输出(1个值)， A 为 行为输出(多个action值)
    # 为了避免 V = 0，所以 A = A - avg (A)  A 所有行为的均值 
    out = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)

#### A3C 并行学习，提升速度

    使用并行策略，提高学习速度
    
    