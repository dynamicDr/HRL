import math
import pprint
import time

import gym
from gym import wrappers

import rsoccer_gym

# Using VSS Single Agent env
env = gym.make('VSSMA-v0')
# env = wrappers.RecordVideo(env, './videos/' + str(time.time()) + '/')
# Run for 1 episode and print reward at the end
v_list = []
max_step=1000
for i in range(10):
    env.reset()
    done = False
    episode_step=0
    while not done and episode_step<max_step:
        episode_step+=1
        # Step using random actions
        action = [[1,1],[0,0],[0,0]]
        # action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # pprint.pprint(env.individual_reward['robot_0'])
        # print(reward['robot_0'])
        # print(next_state)
        print(env.field)
        # pprint.pprint(reward)
        # time.sleep(1)
        ball_x = next_state[0][0]
        ball_y = next_state[0][1]
        print(f"{ball_x},{ball_y}")

        env.render()

#print(v_list)
# plt.scatter(range(len(v_list)), v_list, s=1)
# plt.show()