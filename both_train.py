import argparse
import copy
import pickle
import re

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from Timer import Timer
from coach_mmoe import Coach_MMOE
from matd3 import MATD3
from replay_buffer import ReplayBuffer, CoachReplayBuffer
from rsoccer_gym.vss.env_ma import VSSMAEnv
from rsoccer_gym.vss.env_ma.vss_gym_ma import VSSMABoth


class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Create env
        self.env: VSSMABoth = gym.make(self.env_name)
        self.args.N = 3  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space.shape[1] for i in range(self.args.N)]  # 46
        self.args.action_dim_n = [self.env.action_space.shape[1] for i in range(self.args.N)]
        self.args.coach_obs_dim = 40
        self.args.coach_action_dim = 2 * args.N
        print(f"action_dim={self.args.action_dim_n}")
        print(f"obs_dim={self.args.obs_dim_n}")

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create a tensorboard
        self.writer = None
        if not self.args.display:
            self.writer = SummaryWriter(
                log_dir='runs/{}/number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        # Create N agents and coach
        self.agent_n = [MATD3(args, agent_id, self.writer) for agent_id in range(args.N)]
        self.coach = Coach_MMOE(args, self.writer)
        self.opp_agent_n = [MATD3(args, agent_id, self.writer) for agent_id in range(args.N)]

        # Load pre-trained model
        for i in range(len(self.agent_n)):
            self.agent_n[i].actor.load_state_dict(torch.load(args.team_blue_agent.format(i)))
            self.opp_agent_n[i].actor.load_state_dict(torch.load(args.team_yellow_agent.format(i)))
        self.coach.load_model(self.args.mmoe_model_load_path)

        self.replay_buffer = ReplayBuffer(self.args)
        self.coach_replay_buffer = CoachReplayBuffer(self.args)
        self.opp_replay_buffer = ReplayBuffer(self.args)

        if self.args.restore:
            self.total_steps = self.args.restore_step
            self.episode = self.args.restore_episode
        else:
            self.total_steps = 0
            self.episode = 0

        if self.args.display:
            self.noise_std = 0
        elif self.args.restore:
            self.noise_std = self.args.noise_std_min
        else:
            self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self):
        while self.episode < self.args.max_episode:
            # For each episode..
            obs = self.env.reset()
            coach_obs = obs[-1]
            terminate = False
            done = False
            episode_step = 0
            episode_reward = 0
            goal_step = 0
            # Give a initial goal for this episode
            goal = self.coach.choose_action(coach_obs)
            self.env.set_attacker_and_goal(goal)
            goal_init_obs = coach_obs
            agent_obs_n = self.env.observation[:-1]
            opp_obs = copy.deepcopy(self.env.opp_obs)
            while not (done or terminate):
                # For each step...
                a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in
                       zip(self.agent_n, agent_obs_n)]
                obs_next, r_n, done, info = self.env.step(copy.deepcopy(a_n))
                opp_a_n = info["opp_a_n"]
                opp_agent_r_n = info["opp_agent_r_n"]
                opp_obs_next = copy.deepcopy(self.env.opp_obs)
                reward_list = list(r_n.values())
                agent_r_n = reward_list
                if self.args.display:
                    self.env.render()
                if self.args.record_reward and not self.args.display:
                    self.env.write_log(self.writer, self.total_steps)
                agent_obs_next_n = obs_next[:-1]
                # Store the transition
                self.replay_buffer.store_transition(agent_obs_n, a_n, agent_r_n, agent_obs_next_n, done)
                self.opp_replay_buffer.store_transition(opp_obs, opp_a_n, opp_agent_r_n, opp_obs_next, done)

                obs = obs_next
                opp_obs = opp_obs_next
                coach_obs = obs_next[-1]
                agent_obs_n = obs_next[:-1]
                self.total_steps += 1
                episode_step += 1
                goal_step += 1
                episode_reward += sum(r_n.values())

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.replay_buffer.current_size > self.args.batch_size and not self.args.display:
                    # Train each agent individually
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

                if episode_step >= self.args.episode_limit:
                    terminate = True

                # Update the goal
                if goal_step == self.args.goal_update_freq or (terminate or done):
                    self.coach_replay_buffer.store_transition(goal_init_obs, coach_obs)
                    if not (terminate or done):
                        # give a new goal
                        goal = self.coach.choose_action(coach_obs)
                        self.env.set_attacker_and_goal(goal)
                        goal_init_obs = coach_obs
                        agent_obs_n = self.env.observation[:-1]
                        goal_step = 0
            self.episode += 1

            if self.coach_replay_buffer.current_size >= self.args.coach_batch_size and not self.args.display:
                # Train coach
                self.coach.train(self.coach_replay_buffer, self.total_steps)

            # Save model and For selfplay:update opponent
            if self.episode % self.args.save_rate == 0 and not self.args.display:
                for i in range(args.N):
                    self.agent_n[i].save_model(number, self.total_steps, i)
                    if self.args.self_play:
                        self.agent_n[i].save_model(number, self.total_steps, i,self.args.opp_path, save_as_opp=True)
                self.coach.save_model(number, self.total_steps,self.args.mmoe_model_save_path)

                if self.args.self_play:
                    self.coach.save_model(number, self.total_steps, self.args.opp_path, save_as_opp = True)
                    self.env.load_opp()

            avg_train_reward = episode_reward / episode_step
            print("============epi={},step={},avg_reward={},goal_score={}==============".format(self.episode,
                                                                                                self.total_steps,
                                                                                                avg_train_reward,
                                                                                                info["goal_score"]))
            if not self.args.display:
                self.writer.add_scalar('Agent rewards for each episode', avg_train_reward, global_step=self.episode)
                self.writer.add_scalar('Goal', info["goal_score"], global_step=self.episode)
        self.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_episode", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=300, help="Maximum number of steps per episode")  # 300
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")
    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--save_rate", type=int, default=1000,
                        help="Model save per n episode")
    parser.add_argument("--record_reward", type=bool, default=True, help="Record detailed reward to tensorboard")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")
    parser.add_argument("--display", type=bool, default=True, help="Display mode")
    # ------------------------------------- HRL-------------------------------------------------------------------
    parser.add_argument("--coach_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--coach_max_action", type=float, default=1.2, help="Max action")
    parser.add_argument("--goal_update_freq", type=int, default=30, help="The frequency of coach giving a new goal")
    parser.add_argument("--lr_mmoe", type=float, default=1e-4, help="Learning rate of mmoe")
    parser.add_argument("--coach_buffer_size", type=int, default=int(3e3), help="The capacity of the replay buffer")
    parser.add_argument("--coach_batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--mmoe_model_save_path", type=str,
                        default="./models/coach/")
    # ------------------------------------- Both-train------------------------------------------------------------
    parser.add_argument("--team_blue_agent", type=str,
                        default="/home/user/football/HRL/models/agent/actor_number_19_552k_agent_{}.pth")
    parser.add_argument("--mmoe_model_load_path", type=str,
                        default="/home/user/football/HRL/models/coach/moe_num_19_552k")
    parser.add_argument("--team_yellow_agent", type=str,
                        default="/home/user/football/HRL/models/agent/actor_number_19_552k_agent_{}.pth")

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    env_name = "VSSMABoth-v0"
    seed = 0
    number = 20

    runner = Runner(args, env_name=env_name, number=number, seed=seed)

    # Save args
    with open(f'./models/args/args_num{number}.npy', 'wb') as f:
        pickle.dump(runner.args, f)
    if args.self_play and not args.display:
        with open(f'{args.opp_path}args.npy', 'wb') as f:
            pickle.dump(runner.args, f)

    # if args.restore and not args.display:
    #     load_number = re.findall(r"number_(.+?)_", args.restore_model_dir)[0]
    #     assert load_number == str(number)
    #     print("Loading...")
    #     for i in range(len(runner.agent_n)):
    #         runner.agent_n[i].actor.load_state_dict(torch.load(args.restore_model_dir.format(i)))
    #     runner.restore_train()
    print("Start runner.run()")
    runner.run()
