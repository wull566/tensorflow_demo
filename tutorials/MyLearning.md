# 机器学习知识点 Machine Learning (ML) 

## 机器学习概论

#### 分类

    监督学习
    非监督学习
    半监督学习

#### 主要算法

    神经网络:
    强化学习：投篮命中强化
    遗传算法： 通过淘汰弱者进行强化

## 神经网络知识点 Tensorflow (TF)

#### 激励函数

    线性方程 y = Wx 和 非线性方程 y = AF(Wx)
    激励函数使得线性方程 变成 非线性方程
    注意: 特别多层的神经网络不能随意选择，会涉及梯度爆炸和梯度消失的问题
    
    常用选择:
    sigmoid, relu, tanh
    卷积神经网络推荐，relu
    循环神经网络， tanh 或 relu
    
    激励函数在隐藏层输出时过激励函数变化 activation, 隐藏节点达到阈值激活
        
#### 训练优化器 Optimizer
    
    SGD：分批分量训练
    Momentum: 增加下坡惯性 W = b1 * W - Learning rate * dx   原:  W = - Learning rate * dx
    AdaGrad: 增加错误方向阻力
    RMSProp: Momentum + AdaGrad 缺少部分参数
    Adam: 完美结合 Momentum + AdaGrad  (tf.train.AdamOptimizer)

#### 过拟合问题

* L1,L2... 算法: 
    
        cost=(Wx - realy)^2 + (W)^2
    
* dropout 神经网络过拟合算法: 随机忽略部分隐藏层

#### Saver保存和读取
    
    使用 tf.train.Saver() 组件来保存数据
    
    saver.save(sess, 'my_net/save_net.ckpt')        //保存
    saver.restore(sess, 'my_net/save_net.ckpt')     //读取
    
#### tf.data.Dataset 构建数据集
    
    dataset = tf.data.Dataset.from_tensor_slices((tfx, tfy))
    dataset = dataset.shuffle(buffer_size=1000)   # choose data randomly from this buffer
    dataset = dataset.batch(32)                   # batch size you will use
    dataset = dataset.repeat(3)                   # repeat for 3 epochs
    iterator = dataset.make_initializable_iterator()  # later we have to initialize this one
    
    # your network
    bx, by = iterator.get_next()                  # use batch to update


#### 使用 可视化梯度下降 来调整参数

#### 处理不均衡数据

* 获取更均衡数据 (靠谱)
* 换个评判方式
* 重组数据 (靠谱)
* 使用其他学习方法
* 修改算法


#### 批标准化 (BN: Batch Normalization )

    BN 解决了反向传播过程中的梯度问题（梯度消失和爆炸），同时使得不同scale的 w 整体更新步调更一致。
    
    

