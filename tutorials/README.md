# 机器学习实战

## 神经网络 tensorflow

#### CNN 卷积神经网络

    一般用于机器图片视觉，读取图片含义。
    例如: 辨别图片描述动物，辨别手写数字
    
    离散型图片分类器。

#### RNN 循环神经网络

    一般用于连续动作行为，有上下文关系。

    根据语境和之前预测的数据，预测之后的数据含义。
    例如: 用语句描述图片含义、 中英文语句翻译、 电脑作曲
    
    LSTM 长短期记忆 cell
    主线支线，通过三个门，判断主支线平衡 (writeGate, ForgetGate, ReadGate)
    
#### AutoEncoder 自编码（非监督学习)

    作用: 上千万高清图片压缩一下, 提取图片中的最具代表性的信息,
    再把缩减过后的信息放进神经网络学习.再还原图片数据
    
    类似 PCA 主成分分析，数据降维
    
## * GAN 生成对抗网络（非监督学习)

    新手画家 + 新手鉴赏家 = 随机生成名画
    
    
    GAN, GAN-CLS, GAN-INT,GAN-INT-CLS
    
#### * CGAN 有条件的生成对抗网络 

    
    

## 强化学习

#### Q Learning 和 Sarsa 传统强化学习对比：
 
* Q Learning 在行为收益计算时只关心行为最大收益，即达到目的的收益，不考虑其他行为的风险

* Sarsa 计算收益时，会去反馈下一行为概率的所有可能的风险和收益

* Sarsa(lambda) 回合衰变更新方法， lambda 衰变强度 0 - 1 

#### DQN : Deep Q Network  神经网络强化学习

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


#### Policy Gradients 基于策略梯度下降算法

    PG支持连续性行为，如开车, 以回合性质学习    
    * 随机一回合的用户行为列表，然后根据奖惩和对结果的影响率对行为评价，
    放大或缩小该行为在下次选择中的概率
    
    DQN是以离散化行为进行分析选择
    
    
    观测信息通过神经网络分析, 选左边的行为, 直接进行反向传递, 使之下次被选的可能性增加, 
    但是奖惩信息却告诉我们, 这次的行为是不好的, 那我们的动作可能性增加的幅度 随之被减低.
        
    每一个回合去学习，这一个回合的所有行为对与结果的奖惩优化
    
    
#### Actor Critic 动作评论家（失效）

    Actor Critic = Q learning + Policy Gradients 
    
    Policy Gradients: 随机用户行为
    Q learning：评论动作数据，每步更新
    
    问题：连续动作导致学不到东西
        
#### * DDPG： Deep Deterministic Policy Gradient 动作评论家可用版

    Deep Deterministic Policy Gradient = Actor Critic + DQN
    Actor Critic = Policy Gradients + 评价(DQN)
    先用PG概率化选择动作，然后用DQN对动作进行评价，然后用反馈机制来优化PG选择的动作
    
    每次行为优化，非回合制
    
    
    Deep: 深度
    Deterministic: 确定输出一个动作
    Policy Gradient: 在连续动作中选择动作Actor
    
    问：练习代码中，并非需要估计现实双网络并行？
    
    
#### * A3C 并行学习，提升速度

    使用并行策略，提高学习速度。
    使用多场景并行同时计算，提高学习速度。
    解决单一学习模式下，梯度下降单一，非全局最优的问题
    
#### * DPPO (Distributed Proximal Policy Optimization) 分布式近端策略优化

    PPO: 解决 Policy Gradient 不好确定 Learning rate, 如果太大不收敛， 太小训练太慢。
    Importance Sampling: 重要性抽样
    
    
##  进化学习 Evolutionary

#### 遗传算法

     根据进化论推演的算法, 定义DNA, 种群大小，父母DNA交叉配对，概率变异，
     自然选择，适者生存，概率越大，被选中的次数越多

#### 微生物遗传算法

    对比两个个体，产生loser和winner, 失败者和胜利者交配变异，去除失败者
    
    在袋子里抽两个球, 对比两个球, 把球大的放回袋子里, 把球小的变一下再放回袋子里

#### 进化策略

    DNA和变异强度，父母DNA配对，根据变异强度对每个DNA值进行正态分布变异，变异强度慢慢收敛,
    所有数据统一排序取最佳数据集

    DNA： 实数代替01结构
    变异： 父母变异均值，根据每个值的变异强度，进行正态分布概率获取 
      
#### (1+1)-ES 进化

    父子节点竞争，子节点变异，子节点优胜扩大变异，否则收敛变异。
    
    降低集群竞争为两节点竞争，降低复杂度

#### NES 自然进化策略 (tf)

    NES: 生宝宝, 用好宝宝的梯度辅助找到前进的方向
    使用 tensorflow 中 MultivariateNormalFullCovariance 计算多元正态分布
    并且使用梯度下降方法找到进化方向
    
#### 神经网络进化策略

    遗传算法 + 进化策略 + 神经网络

    NEAT算法: 使用遗传算法进化神经网络节点的结构，实现网络自我进化
    
#### ES: 进化策略强化学习

    ES : 在自己附近生宝宝, 让自己更像那些表现好的宝宝
    
    ES: 缺点，性能消耗较大
    
#### CMA-ES 协方差矩阵自适应进化策略

    使用最佳解的信息来调整其均值和协方差矩阵，所以在距离最佳解很远时，
    它可以决定扩大至更广泛的网络，或者当距离最佳解很近时，缩小搜索空间

##  迁移学习 TransferLearning

    利用已经完成训练算法的神经网络数据的基础上，嫁接计算另外的神经网络
    
    如 男女 迁移的 男童，女童
    注意不能迁移差别较大的两个网络，会有先入为主的想法，不如重新学习。