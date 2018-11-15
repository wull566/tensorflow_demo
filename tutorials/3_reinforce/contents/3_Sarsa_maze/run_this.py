"""
Sarsa is a online updating method for Reinforcement 1_tensorflow_new.

Unlike Q 1_tensorflow_new which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q 1_tensorflow_new is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # 先选择行为
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # 移动
            observation_, reward, done = env.step(action)

            # 再次选择行为
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break
        print("============", episode)
        print(RL.q_table)

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()