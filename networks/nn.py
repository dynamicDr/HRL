# -*- coding: utf-8 -*-
# @Time    : 2021-04-19 12:12
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  script description

import torch
import torch.nn as nn


class NN(nn.Module):
    """
    MMOE for CTCVR problem
    """
    def __init__(self, n_expert=3, mmoe_hidden_dim=128,
                 hidden_dim=[128, 64], dropouts=[0.5, 0.5], output_size=36, expert_activation=None, num_task=1):
        """
        MMOE model input parameters
        :param feature_dict: feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param emb_dim: int embedding dimension
        :param n_expert: int number of experts in mmoe
        :param mmoe_hidden_dim: mmoe layer input dimension
        :param hidden_dim: list task tower hidden dimension
        :param dropouts: list of task dnn drop out probability
        :param output_size: int task output size
        :param expert_activation: activation function like 'relu' or 'sigmoid'
        :param num_task: int default 2 multitask numbers
        """
        super(NN, self).__init__()
        self.expert_activation = expert_activation

        # expert1
        self.nn = torch.nn.Sequential()
        self.nn.add_module('nn_layer_1_linear', nn.Linear(in_features=40*1, out_features=128*1))
        self.nn.add_module('nn_layer_2_linear', nn.Linear(128, 128))
        self.nn.add_module('nn_layer_3_linear', nn.Linear(128, 128))
        self.nn.add_module('nn_layer_4_linear', nn.ReLU())
        self.nn.add_module('nn_layer_5_linear', nn.Linear(128, 128))
        self.nn.add_module('nn_layer_6_batchnorm', nn.BatchNorm1d(128))
        self.nn.add_module('nn_layer_7_dropout', nn.Dropout(0.5))
        self.nn.add_module('nn_layer_last_linear', nn.Linear(128, output_size))


        # # expert2
        # self.blue_expert = torch.nn.Sequential()
        # self.blue_expert.add_module('blue_expert_layer_1_linear', nn.Linear(in_features=21*1, out_features=128*1))
        # self.blue_expert.add_module('blue_expert_layer_2_linear', nn.Linear(128, 128))
        # self.blue_expert.add_module('blue_expert_layer_3_linear', nn.Linear(128, 128))
        #
        #
        # # expert3
        # self.yellow_expert = torch.nn.Sequential()
        # self.yellow_expert.add_module('yellow_expert_layer_1_linear', nn.Linear(in_features=15*1, out_features=128*1))
        # self.yellow_expert.add_module('yellow_expert_layer_2_linear', nn.Linear(128, 128))
        # self.yellow_expert.add_module('yellow_expert_layer_3_linear', nn.Linear(128, 128))

        # self.yellow_expert.add_module('yellow_expert_layer_4_linear', nn.Linear(128, 64))
        # self.yellow_expert.add_module('yellow_expert_layer_5_linear', nn.Linear(64, 64))
        # self.yellow_expert.add_module('yellow_expert_layer_6_linear', nn.Linear(64, 64))

        # gates
        # self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in range(num_task)]
        # for gate in self.gates:
        #     gate.data.normal_(0, 1)
        # self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]

        # self.gate = torch.nn.Sequential()
        # self.gate.add_module('gate_layer_1_liner', nn.Linear(in_features=40*1, out_features=3*1))


        # self.output_layers = torch.nn.Sequential()
        # self.output_layers.add_module('output_layer_1_linear', nn.Linear(128, 128))
        # self.output_layers.add_module('output_layer_2_batchnorm', nn.BatchNorm1d(128))
        # self.output_layers.add_module('output_layer_3_dropout', nn.Dropout(0.5))
        # self.output_layers.add_module('output_layer_last_linear', nn.Linear(128, output_size))

    def forward(self, x):
        # expert_out = torch.relu(self.expert(x.float()).unsqueeze(dim=2))#[5,128,1]
        # expert2_out = torch.relu(self.blue_expert(x[:,4:25].float()).unsqueeze(dim=2))#[5,128,1]
        # expert3_out = torch.relu(self.yellow_expert(x[:,25:40].float()).unsqueeze(dim=2))#[5,128,1]
        #
        # experts_out = torch.cat((expert1_out, expert2_out, expert3_out), dim=2)
        #
        # gate_out = nn.Softmax(dim=-1)(self.gate(x.float()))

        # expanded_gate_output = torch.unsqueeze(gate_out, 1)  # batch * 1 * num_experts
        # weighted_expert_output = experts_out * expanded_gate_output.expand_as(experts_out)  # batch * mmoe_hidden_size * num_experts
        # outs = torch.sum(weighted_expert_output, 2)

        # task_outputs = self.output_layers(expert_out)
        task_outputs = self.nn(x.float())  # [5,128,1]

        return task_outputs
  
    
if __name__ == "__main__":
    import numpy as np
    
    a = torch.from_numpy(np.array([[ 0.3949057,  -0.67849123,  0.49404806, -0.8383228,  -0.6695515,  -0.5077701,
                                     -0.76978445,  0.63830394,  0.12520742, -0.30994287,  0.6646415,  -0.5060467,
                                     -0.4852863,  -0.4803488,   0.8770776,  -0.44585156,  0.36836398,  0.42954314,
                                      0.28829348, -0.4717942,  -0.8896054,   0.45672998,  0.44750333, -0.8805927,
                                      0.01134013, -0.24162708, -0.14976601,  0.45704857, -0.20702872, -0.505435,
                                      0.06800497,  0.02031459, -0.1524698,  -0.42110097, -0.5667639,   0.34711796,
                                     -0.587568,   -0.04369975, -0.6794261,  -0.3381719 ],
                                    [ 0.76072884, -0.49609482, -0.18146583,  0.19788463, -0.52517563, -0.59319454,
                                     -0.26532206,  0.96415985,  0.3609187,  -0.21789363,  0.587068,   -0.53371215,
                                     -0.38389462, -0.9871627,   0.1597177,   0.03037784,  0.5568751,   0.45199567,
                                      0.68381864, -0.39801985, -0.9874293,  -0.15806133,  0.17589658,  0.4452042,
                                      0.24473238, -0.1796447,  -0.38816965,  0.33255455,  0.18403183, -0.6198017,
                                      0.20249282, -0.01065929,  0.28127033, -0.49174607, -0.44373816,  0.59301245,
                                     -0.45641357,  0.1455089,   0.21864238,  0.04206641],
                                    [ 0.80087644, -0.5771021,   0.55680776, -0.6255986,  -0.51364875, -0.49042284,
                                     -0.7263025,   0.6873752,   0.18316552, -0.39231184,  0.5150403,  -0.67696285,
                                     -0.3464939,   0.9667131,   0.2558629,  -0.3050388,  -0.4288497,   0.46916562,
                                      0.75079316, -0.48317087,  0.24617377,  0.9692257,   0.5196289,  -0.05499678,
                                      0.49952933, -0.08092439, -0.4533274,  -0.07709455, -0.12646389, -0.8497601,
                                      0.25517833, -0.04402829,  0.32381338, -0.1438022,  -0.65150416,  0.64934236,
                                     -0.4330346,   0.15765761, -0.2430165,  -0.21343268],
                                    [ 0.78373444, -0.6199196,   0.21609896,  0.34822306, -0.42414114, -0.3161109,
                                     -0.27486008, -0.96148425, -0.48188758, -0.01464697,  0.5337263,  -0.5081595,
                                     -0.4418845,  -0.99751014, -0.07052363,  0.14565037,  0.6298312,   0.30518973,
                                      0.6806718,  -0.6034567,   0.00181238, -0.99999833, -0.13973635,  0.06085636,
                                      0.7913867,  -0.07011848, -0.4612662,   0.08607629,  0.27608347, -0.70601934,
                                      0.29796255, -0.16124271,  0.20411418, -0.5748453,  -0.41384077,  0.53950673,
                                     -0.44883254, -0.05137607,  0.70636916, -0.3081357 ],
                                    [ 0.7840396,  -0.5063278,  -0.05320276, -0.06386313, -0.4086613,  -0.19295755,
                                     -0.99729234,  0.07353889,  0.01775181,  0.08053262,  0.6792153,  -0.48190662,
                                     -0.29023734, -0.7453329,   0.6666925,  -0.07881436,  0.13800564,  0.5642232,
                                      0.663418,   -0.537769,   -0.78615254,  0.6180325,   0.0895606,  -0.26976237,
                                      0.6241843,  -0.00521955, -0.49396595, -0.0745832,  -0.21756338, -0.7662257,
                                      0.35264766, -0.23141591,  0.02803912, -0.45999703, -0.56533945,  0.70015305,
                                     -0.30733666,  0.35172808,  0.0203599,  -0.624249  ]]))
    feature_dict = {'Ball_X': (1, 0), 'Ball_Y': (1, 1), 'Ball_Vx': (1, 2), 'Ball_Vy': (1, 3), 'id_0_Blue_Robot_X': (1, 4), 'id_0_Blue_Robot_Y': (1, 5), 'id_0_Blue_Robot_sin(theta)': (1, 6), 'id_0_Blue_Robot_cos(theta)': (1, 7), 'id_0_Blue_Robot_Vx': (1, 8), 'id_0_Blue_Robot_Vy': (1, 9), 'id_0_Blue_Robot_v_theta': (1, 10), 'id_1_Blue_Robot_X': (1, 11), 'id_1_Blue_Robot_Y': (1, 12), 'id_1_Blue_Robot_sin(theta)': (1, 13), 'id_1_Blue_Robot_cos(theta)': (1, 14), 'id_1_Blue_Robot_Vx': (1, 15), 'id_1_Blue_Robot_Vy': (1, 16), 'id_1_Blue_Robot_v_theta': (1, 17), 'id_2_Blue_Robot_X': (1, 18), 'id_2_Blue_Robot_Y': (1, 19), 'id_2_Blue_Robot_sin(theta)': (1, 20), 'id_2_Blue_Robot_cos(theta)': (1, 21), 'id_2_Blue_Robot_Vx': (1, 22), 'id_2_Blue_Robot_Vy': (1, 23), 'id_2_Blue_Robot_v_theta': (1, 24), 'id_0_Yellow_Robot_X': (1, 25), 'id_0_Yellow_Robot_Y': (1, 26), 'id_0_Yellow_Robot_Vx': (1, 27), 'id_0_Yellow_Robot_Vy': (1, 28), 'id_0_Yellow_Robot_v_theta': (1, 29), 'id_1_Yellow_Robot_X': (1, 30), 'id_1_Yellow_Robot_Y': (1, 31), 'id_1_Yellow_Robot_Vx': (1, 32), 'id_1_Yellow_Robot_Vy': (1, 33), 'id_1_Yellow_Robot_v_theta': (1, 34), 'id_2_Yellow_Robot_X': (1, 35), 'id_2_Yellow_Robot_Y': (1, 36), 'id_2_Yellow_Robot_Vx': (1, 37), 'id_2_Yellow_Robot_Vy': (1, 38), 'id_2_Yellow_Robot_v_theta': (1, 39)}
    mmoe = MMOE()
    outs = mmoe(a)
    #print(np.zeros(outs.shape))
    top1 = np.zeros(outs.shape)
    top2 = np.zeros(outs.shape)#(5,36)
    #print(outs)
    #print(torch.topk(outs, 2, dim=1))
    _, indices = torch.topk(outs, 2, dim=1)
    #print(indices)
    index_top1 = indices[:,0]
    for i,index in enumerate(indices):
        top1[i][index[0]] = 1
        top2[i][index[1]] = 1
    print(top1.shape)
    print(('==============='))
    print(top2)



    # print(torch.topk(outs[0],2))
    # _, indices = torch.topk(outs[0], 2)
    # print('==========')
    # # print(indices)
    # top1 = np.zeros((36,))
    # top2 = np.zeros((36,))
    # top1[indices[0]] = 1
    # top2[indices[1]] = 1
    # print(top1)
    # print(top2)


