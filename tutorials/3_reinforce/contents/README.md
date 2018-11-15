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


## Policy Gradients  方案渐变 强化选项

    不以误差反向传递，更加奖惩reword传递。
    DQN是以离散化行为进行分析选择
    PG支持连续性行为，如开车, 以回合性质学习    
    
    观测信息通过神经网络分析, 选左边的行为, 直接进行反向传递, 使之下次被选的可能性增加, 
    但是奖惩信息却告诉我们, 这次的行为是不好的, 那我们的动作可能性增加的幅度 随之被减低.
        
    每一个回合去学习，这一个回合的所有行为对与结果的奖惩优化
    
    
#### Actor Critic 动作评论家（失效）

    Actor Critic = Q learning + Policy Gradients 
    
    Policy Gradients: 
    Q learning：评论动作数据，每步更新
    
    问题：连续动作导致学不到东西
        
#### Deep Deterministic Policy Gradient （DDPG）动作评论家可用版

    Deep Deterministic Policy Gradient = Actor Critic + DQN
    
    Deep: 更深层次的
    Deterministic: 确定输出一个动作
    Policy Gradient: 在连续动作中选择动作Actor
    
    问：练习代码中，并非需要估计现实双网络并行？
    
    
#### A3C 并行学习，提升速度

    使用并行策略，提高学习速度
    
#### DPPO (Distributed Proximal Policy Optimization) 分布式近端策略优化

    PPO: 解决 Policy Gradient 不好确定 Learning rate, 如果太大不收敛， 太小训练太慢。
    Importance Sampling: 重要性抽样
    