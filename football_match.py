import copy
import datetime
import os
import pickle

import gym
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter

from coach_mmoe import Coach_MMOE
from match_result.plot import match_plot
from matd3 import MATD3
from replay_buffer import CoachReplayBuffer
from rsoccer_gym.vss.env_ma import VSSMAOpp

# =====================================================

max_match = 100
max_match_step = 1000
match_number = 26
seed = 0
display = False
record_reward = True

online_training = False
online_training_freq = 1  # train 1 batch per n step
online_training_lr = 1e-4
save_rate = 100  # save online trained coach per n episode

multi_attacker_mode = 3

# =====================================================

if display:
    online_training = False

env: VSSMAOpp = gym.make('VSSMAOpp-v0')
env.multiple_attacker_mode = multi_attacker_mode

# Set random seed
np.random.seed(seed)
torch.manual_seed(seed)

# Create a tensorboard
# if not display:
#     writer = SummaryWriter(
#         log_dir='runs/match/match_num_{}_seed_{}'.format(match_number, seed))
# else:
writer = None

# Load players
mmoe_model_load_path = "models/coach/moe_num_6_4094k"
agent_model_load_path = "models/agent/actor_number_6_4094k_agent_{}.pth"
args_load_path = "models/args/args_num6.pkl"

with open(args_load_path, 'rb') as f:
    args = pickle.load(f)
    print(args)
agent_n = [MATD3(args, agent_id, writer) for agent_id in range(args.N)]
coach = Coach_MMOE(args, writer)
coach.load_model(mmoe_model_load_path)
if online_training:
    coach.reset_lr(online_training_lr)
for i in range(args.N):
    agent_n[i].actor.load_state_dict(torch.load(agent_model_load_path.format(i)))
env.load_opp()

# Create replay buffer
coach_replay_buffer = CoachReplayBuffer(args)
match = 0

df_column_names = ["match", "blue_score", "yellow_score", "blue_robot_0_possession_frame",
                   "blue_robot_1_possession_frame", "blue_robot_2_possession_frame", "yellow_robot_0_possession_frame",
                   "yellow_robot_1_possession_frame", "yellow_robot_2_possession_frame",
                   "blue_robot_0_intercept_time", "blue_robot_1_intercept_time", "blue_robot_2_intercept_time",
                   "yellow_robot_0_intercept_time", "yellow_robot_1_intercept_time", "yellow_robot_2_intercept_time",
                   "blue_robot_0_pass_time", "blue_robot_1_pass_time", "blue_robot_2_pass_time",
                   "yellow_robot_0_pass_time", "yellow_robot_1_pass_time", "yellow_robot_2_pass_time",
                   "ball_in_blue_half_frame", "ball_in_yellow_half_frame"]
df = pd.DataFrame(columns=df_column_names)

for match in range(max_match):
    match_step = 0

    match_dict = {}
    for name in df_column_names:
        match_dict[name] = 0
    match_dict["match"] = match

    while match_step < max_match_step:
        print(f"match:{match},step:{match_step}")

        # For each episode..
        obs = env.reset()
        coach_obs = obs[-1]
        terminate = False
        done = False
        episode_step = 0
        episode_reward = 0
        goal_step = 0

        # Give a initial goal for this episode
        goal = coach.choose_action(coach_obs)
        env.set_attacker_and_goal(goal)
        goal_init_obs = coach_obs
        agent_obs_n = env.observation[:-1]
        while not (done or terminate):
            # For each step...
            a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in
                   zip(agent_n, agent_obs_n)]
            obs_next, r_n, done, info = env.step(copy.deepcopy(a_n))
            match_step += 1
            if match_step >= max_match_step:
                terminate = True
            # kpi
            robot_idx = info["possession_robot_idx"]
            if info["possession_team"] == 0:
                match_dict[f"blue_robot_{robot_idx}_possession_frame"] += 1
            elif info["possession_team"] == 1:
                match_dict[f"yellow_robot_{robot_idx}_possession_frame"] += 1

            if info["intercept"] == 1:
                match_dict[f"blue_robot_{robot_idx}_intercept_time"] += 1
            elif info["intercept"] == -1:
                match_dict[f"yellow_robot_{robot_idx}_intercept_time"] += 1

            if info["passing"] == 1:
                match_dict[f"blue_robot_{robot_idx}_pass_time"] += 1
            elif info["passing"] == -1:
                match_dict[f"yellow_robot_{robot_idx}_pass_time"] += 1

            if obs_next[0][0] > 0:
                match_dict["ball_in_blue_half_frame"] += 1
            elif obs_next[0][0] < 0:
                match_dict["ball_in_yellow_half_frame"] += 1

            reward_list = list(r_n.values())
            agent_r_n = reward_list
            if display:
                env.render()
            agent_obs_next_n = obs_next[:-1]
            obs = obs_next
            coach_obs = obs_next[-1]
            agent_obs_n = obs_next[:-1]
            goal_step += 1
            episode_reward += sum(r_n.values())

            # Update the goal
            if goal_step == args.goal_update_freq or (terminate or done):
                if online_training:
                    coach_replay_buffer.store_transition(goal_init_obs, coach_obs)
                if not (terminate or done):
                    # give a new goal
                    goal = coach.choose_action(coach_obs)
                    env.set_attacker_and_goal(goal)
                    goal_init_obs = coach_obs
                    agent_obs_n = env.observation[:-1]
                    goal_step = 0
        if info["goal_score"] == 1:
            match_dict["blue_score"] += 1
        elif info["goal_score"] == -1:
            match_dict["yellow_score"] += 1

    df = df.append(pd.Series(match_dict), ignore_index=True)
env.close()

folder = f"match_result/{match_number}_{datetime.datetime.now()}"
os.mkdir(folder)
match_plot(df,folder)

df.to_csv(f"{folder}/{match_number}.csv",index=False)