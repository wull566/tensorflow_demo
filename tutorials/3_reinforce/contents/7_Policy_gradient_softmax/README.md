# 强化学习

## Policy Gradients  方案渐变 强化选项

    不以误差反向传递，更加奖惩reword传递。
    DQN是以离散化行为进行分析选择
    PG支持连续性行为，如开车, 以回合性质学习    
    
    观测信息通过神经网络分析, 选左边的行为, 直接进行反向传递, 使之下次被选的可能性增加, 
    但是奖惩信息却告诉我们, 这次的行为是不好的, 那我们的动作可能性增加的幅度 随之被减低.
        
    每一个回合去学习，这一个回合的所有行为对与结果的奖惩优化



    