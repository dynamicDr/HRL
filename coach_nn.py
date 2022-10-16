import numpy as np
import torch
import torch.nn as nn
from networks.mmoe import MMOE
from networks.nn import NN
from replay_buffer import CoachReplayBuffer


class Coach_NN(object):
    def __init__(self, args, writer):
        self.writer = writer
        self.nn_net = NN()
        self.nn_net.eval()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.nn_net.parameters(), lr=args.lr_mmoe)

    def reset_lr(self,lr):
        self.optimizer = torch.optim.Adam(self.nn_net.parameters(), lr=lr)

    def choose_action(self, obs):
        self.nn_net.eval()
        outs = self.nn_net(torch.from_numpy(np.array([obs])))
        _, indices = torch.topk(outs, 2, dim=1)
        top_1 = indices[0][0]
        top_2 = indices[0][1]
        top_1_x = top_1 % 6
        top_1_y = int(top_1 / 6)
        top_2_x = top_2 % 6
        top_2_y = int(top_2 / 6)

        top_1_x_pos = -0.75 + 1.5 / 6 / 2 + top_1_x * 1.5 / 6
        top_1_y_pos = -0.65 + 1.3 / 6 / 2 + top_1_y * 1.3 / 6

        top_2_x_pos = -0.75 + 1.5 / 6 / 2 + top_2_x * 1.5 / 6
        top_2_y_pos = -0.65 + 1.3 / 6 / 2 + top_2_y * 1.3 / 6
        # print(f"top1:{top_1}=[{top_1_x},{top_1_y}]=[{top_1_x_pos},{top_1_y_pos}]")
        # print(f"top2:{top_2}=[{top_2_x},{top_2_y}]=[{top_2_x_pos},{top_2_y_pos}]")
        return np.array([[top_1_x_pos, top_1_y_pos], [top_2_x_pos, top_2_y_pos]])

    def train(self, replay_buffer: CoachReplayBuffer, step):
        self.nn_net.train()
        batch_obs_n, batch_tag_n = replay_buffer.sample()
        batch_size = replay_buffer.batch_size
        predict = self.nn_net(batch_obs_n)
        one_index = []
        for i in range(batch_size):
            one_index.append(int(batch_tag_n[i][0] + batch_tag_n[i][1] * 6))
        one_index = torch.tensor(one_index)
        loss = self.loss_function(predict, one_index)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.writer is not None:
            self.writer.add_scalar('MMOE loss', loss, global_step=step)

    def load_model(self, model_path):
        file = torch.load(model_path)
        if type(file) == MMOE:
            self.nn_net.load_state_dict(torch.load(model_path).state_dict())
        else:
            self.nn_net.load_state_dict(torch.load(model_path))
        print(f"Successfully load mmoe model. model_path:{model_path}")

    def save_model(self, number, total_steps,save_path, online_training = False, save_as_opp = False):
        if online_training:
            torch.save(self.nn_net.state_dict(), f"{save_path}online_trained_moe_num_{number}_{int(total_steps / 1000)}k")
        elif save_as_opp:
            torch.save(self.nn_net.state_dict(), f"{save_path}coach")
        else:
            torch.save(self.nn_net.state_dict(), f"{save_path}moe_num_{number}_{int(total_steps / 1000)}k")
