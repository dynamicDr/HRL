import copy
import pickle

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from coach_mmoe import Coach_MMOE
from matd3 import MATD3
from replay_buffer import CoachReplayBuffer
from rsoccer_gym.vss.env_ma import VSSMAOpp



# =====================================================

max_episode = 500
match_number = 8
seed = 0
display = False
record_reward = True

online_training = True
online_training_freq = 1 # train 1 batch per n step
online_training_lr = 1e-4
save_rate = 100 # save online trained coach per n episode

multi_attacker_mode = False

# =====================================================

if display:
    online_training = False

env: VSSMAOpp = gym.make('VSSMAOpp-v0')
env.multiple_attacker_mode = multi_attacker_mode

# Set random seed
np.random.seed(seed)
torch.manual_seed(seed)

# Create a tensorboard
if not display:
    writer = SummaryWriter(
        log_dir='runs/match/match_num_{}_seed_{}'.format(match_number, seed))
else:
    writer = None

# Load players
mmoe_model_load_path = "models/coach/moe_num_16_2520k"
agent_model_load_path =  "models/agent/actor_number_16_2520k_agent_{}.pth"
args_load_path = "models/args/args_num16.npy"
with open(args_load_path, 'rb') as f:
    args = pickle.load(f)
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

total_steps = 0
episode = 0
goal_count = 0

while episode < max_episode:
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
        reward_list = list(r_n.values())
        agent_r_n = reward_list
        if display:
            env.render()
        if record_reward and writer is not None:
            env.write_log(writer,total_steps)
        agent_obs_next_n = obs_next[:-1]
        obs = obs_next
        coach_obs = obs_next[-1]
        agent_obs_n = obs_next[:-1]
        total_steps += 1
        episode_step += 1
        goal_step += 1
        episode_reward += sum(r_n.values())

        if episode_step >= args.episode_limit:
            terminate = True

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
                goal_count += 1
    episode += 1

    # Do online training
    if total_steps % online_training_freq == 0 and coach_replay_buffer.current_size >= args.coach_batch_size and online_training:
        coach.train(coach_replay_buffer, total_steps)

    if episode % save_rate == 0 and not display:
        coach.save_model(match_number, total_steps, "/home/user/football/HRL/models/coach/", online_training=True)

    avg_train_reward = episode_reward / episode_step
    print("============match={},step={},avg_reward={},goal_score={}==============".format(episode,
                                                                                        total_steps,
                                                                                        avg_train_reward,
                                                                                        info["goal_score"]))
    if writer is not None:
        writer.add_scalar('Agent rewards for each episode', avg_train_reward, global_step=episode)
        writer.add_scalar('Goal', info["goal_score"], global_step=episode)
env.close()
