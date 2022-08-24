import numpy as np
import torch
import torch.nn as nn
from networks.mmoe import MMOE
from replay_buffer import CoachReplayBuffer


class Coach_MMOE(object):
    def __init__(self, args, writer):
        self.writer = writer
        self.mmoe_net = MMOE()
        self.mmoe_net.eval()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.mmoe_net.parameters(), lr=args.lr_mmoe)

    def reset_lr(self,lr):
        self.optimizer = torch.optim.Adam(self.mmoe_net.parameters(), lr=lr)

    def choose_action(self, obs):
        self.mmoe_net.eval()
        outs = self.mmoe_net(torch.from_numpy(np.array([obs])))
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
        self.mmoe_net.train()
        batch_obs_n, batch_tag_n = replay_buffer.sample()
        batch_size = replay_buffer.batch_size
        predict = self.mmoe_net(batch_obs_n)
        one_index = []
        for i in range(batch_size):
            one_index.append(int(batch_tag_n[i][0] + batch_tag_n[i][1] * 6))
        one_index = torch.tensor(one_index)
        loss = self.loss_function(predict, one_index)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('MMOE loss', loss, global_step=step)

    def load_model(self, model_path):
        self.mmoe_net.load_state_dict(torch.load(model_path))
        print(f"Successfully load mmoe model. model_path:{model_path}")

    def save_model(self, number, total_steps,save_path, online_training = False, save_as_opp = False):
        if online_training:
            torch.save(self.mmoe_net.state_dict(), f"{save_path}online_trained_moe_num_{number}_{int(total_steps / 1000)}k")
        elif save_as_opp:
            torch.save(self.mmoe_net.state_dict(), f"{save_path}coach")
        else:
            torch.save(self.mmoe_net.state_dict(), f"{save_path}moe_num_{number}_{int(total_steps / 1000)}k")

if __name__ == '__main__':
    # a = [0.3949057, -0.67849123, 0.49404806, -0.8383228, -0.6695515, -0.5077701,
    #      -0.76978445, 0.63830394, 0.12520742, -0.30994287, 0.6646415, -0.5060467,
    #      -0.4852863, -0.4803488, 0.8770776, -0.44585156, 0.36836398, 0.42954314,
    #      0.28829348, -0.4717942, -0.8896054, 0.45672998, 0.44750333, -0.8805927,
    #      0.01134013, -0.24162708, -0.14976601, 0.45704857, -0.20702872, -0.505435,
    #      0.06800497, 0.02031459, -0.1524698, -0.42110097, -0.5667639, 0.34711796,
    #      -0.587568, -0.04369975, -0.6794261, -0.3381719]
    # mmoe = Coach_MMOE(None, None)
    # mmoe.load_model("./mmoe_saved_model/model_mmoe_100")
    # mmoe.choose_action(a)
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    # output.backward()
    print(input)
    print(target)
    print(output)
    # >> >
    # >> >  # Example of target with class probabilities
    # >> > input = torch.randn(3, 5, requires_grad=True)
    # >> > target = torch.randn(3, 5).softmax(dim=1)
    # >> > output = loss(input, target)
    # >> > output.backward()
