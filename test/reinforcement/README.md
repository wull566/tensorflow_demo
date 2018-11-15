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