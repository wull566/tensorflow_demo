"""
Q Learning算法例子
直线找宝藏

Q Learning: 建立一个惩罚奖励Q表，来预测未来行为

"""

import numpy as np
import pandas as pd
import time

# 产生一组伪随机数列，执行一次有效
np.random.seed(2)  # reproducible


N_STATES = 7   # 距离终点多少步 the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # 可选择行为 available actions
EPSILON = 0.9   # 动作选择方法 90%选择最优，10%选择随机动作 greedy police
ALPHA = 0.1     # 学习效率 1_tensorflow_new rate
GAMMA = 0.9    # 衰减度，对未来奖励的衰减 discount factor
MAX_EPISODES = 13   # 训练次数 maximum episodes
FRESH_TIME = 0.3    # 移动速度 fresh time for one move

# 创建Q表
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


# 选择行为
def choose_action(state, q_table):
    # 选择表第几行 如第三行
    state_actions = q_table.iloc[state, :]

    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        # 随机数大于0.9 或 初始化行为状态都零时，选择随机行为
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        # 随机数小于0.9，选择行为表中价值较大的数据
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

# 获得行为对环境的触发，向左移动，向右移动，到达目的
def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

# 刷新绘制环境
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


# 主循环
def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False   # 是否终止符
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # 获得行为状态和行为奖励R
            q_predict = q_table.loc[S, A]   # 获取Q表中的估计值
            if S_ != 'terminal':
                # R 奖励 + 衰减度 *
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            # 奖励更新值
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
        print("第 %s 次:\n" % episode, q_table)
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
