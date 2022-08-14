import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, args):
        self.N = args.N  # The number of agents
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n
        self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n


# class CoachReplayBuffer(object):
#     def __init__(self, args):
#         self.buffer_size = args.coach_buffer_size
#         self.batch_size = args.coach_batch_size
#         self.count = 0
#         self.current_size = 0
#         self.buffer_obs, self.buffer_goal, self.buffer_reward, self.buffer_next_obs = [], [], [], []
#
#         self.buffer_obs=np.empty((self.buffer_size,args.coach_obs_dim))
#         self.buffer_goal=np.empty((self.buffer_size, args.goal_dim))
#         self.buffer_reward=np.empty((self.buffer_size, 1))
#         self.buffer_next_obs=np.empty((self.buffer_size, args.coach_obs_dim))
#
#     def store_transition(self, obs, goal, reward, obs_next):
#         self.buffer_obs[self.count] = obs
#         self.buffer_goal[self.count] = goal
#         self.buffer_reward[self.count] = reward
#         self.buffer_next_obs[self.count] = obs_next
#         self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
#         self.current_size = min(self.current_size + 1, self.buffer_size)
#
#     def sample(self, ):
#         index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
#         batch_obs, batch_goal, batch_reward, batch_obs_next= [], [], [], []
#         batch_obs=torch.tensor(self.buffer_obs[index], dtype=torch.float)
#         batch_goal=torch.tensor(self.buffer_goal[index], dtype=torch.float)
#         batch_reward=torch.tensor(self.buffer_reward[index], dtype=torch.float)
#         batch_obs_next=torch.tensor(self.buffer_next_obs[index], dtype=torch.float)
#
#         return batch_obs, batch_goal, batch_reward, batch_obs_next

class CoachReplayBuffer(object):
    def __init__(self, args):
        self.buffer_size = args.coach_buffer_size
        self.batch_size = args.coach_batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs, self.buffer_tag = [], []

        self.buffer_obs = np.empty((self.buffer_size, args.coach_obs_dim))
        self.buffer_tag = np.empty((self.buffer_size, 2))

    def obs_to_tag(self, obs):
        field_length = 2.4  # 1.5
        field_width = 2.4  # 1.3
        n = 6
        ball_x = obs[0]
        ball_y = obs[1]
        tag_x = int((ball_x + field_width / 2) / (field_width / n))
        tag_y = int((ball_y + field_length / 2) / (field_length / n))
        return [tag_x, tag_y]

    def store_transition(self, init_obs, new_obs):
        self.buffer_obs[self.count] = init_obs
        self.buffer_tag[self.count] = self.obs_to_tag(new_obs)
        self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs, batch_tag = [], []
        batch_obs = torch.tensor(self.buffer_obs[index], dtype=torch.float)
        batch_tag = torch.tensor(self.buffer_tag[index], dtype=torch.float)
        return batch_obs, batch_tag