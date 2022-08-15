import copy
import os.path

import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic_MATD3


class Coach_MATD3(object):
    def __init__(self, args, agent_id, writer):
        self.agent_id = agent_id
        self.max_action = args.coach_max_action
        self.action_dim = args.coach_action_dim
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.goal_update_freq
        self.actor_pointer = 0
        # Create an individual actor and critic for each agent according to the 'agent_id'
        self.actor = Actor(args, -1, True)
        self.critic = Critic_MATD3(args, True)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        self.writer = writer

    # Each agent selects actions based on its own local observations(add noise for exploration)
    def choose_action(self, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        a = self.actor(obs).data.numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def train(self, replay_buffer):
        self.actor_pointer += 1
        batch_obs, batch_goal, batch_reward, batch_obs_next = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            batch_a_next = self.actor_target(batch_obs_next)
            noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            batch_a_next = (batch_a_next + noise).clamp(-self.max_action, self.max_action)

            # Trick 2:clipped double Q-learning
            Q1_next, Q2_next = self.critic_target(batch_obs_next, batch_a_next)
            target_Q = batch_reward[self.agent_id] + self.gamma * torch.min(Q1_next, Q2_next)  # shape:(batch_size,1)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_obs, batch_goal)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.writer.add_scalar('critic_loss_coach'.format(self.agent_id), critic_loss,
                               global_step=self.actor_pointer)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            # Reselect the actions of the agent corresponding to 'agent_id', the actions of other agents remain unchanged
            batch_goal = self.actor(batch_obs)
            actor_loss = -self.critic.Q1(batch_obs, batch_goal).mean()  # Only use Q1
            self.writer.add_scalar('actor_loss_coach'.format(self.agent_id), actor_loss,
                                   global_step=self.actor_pointer)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, env_name, algorithm, number, total_steps, agent_id):
        if not os.path.exists("./model/{}".format(env_name)):
            os.mkdir("./model/{}".format(env_name))
        torch.save(self.actor.state_dict(),
                   "./model/{}/{}_actor_number_{}_step_{}k_coach.pth".format(env_name, algorithm, number,
                                                                             int(total_steps / 1000), agent_id))
