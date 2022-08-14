# -*- coding: utf-8 -*-
# @Time    : 2021-04-19 12:12
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  script description

import torch
import torch.nn as nn


class MMOE(nn.Module):
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
        super(MMOE, self).__init__()
        # check input parameters
        # if feature_dict is None :
        #     raise Exception("input parameter feature_dict must be not None")
        # if isinstance(feature_dict, dict) is False :
        #     raise Exception("input parameter feature_dict must be dict")

        # self.feature_dict = feature_dict
        self.expert_activation = expert_activation
        # self.num_task = num_task

        # user embedding + item embedding
        # hidden_size = len(self.feature_dict)

        # experts
        # self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        # self.experts.data.normal_(0, 1)
        # self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)

        # expert1
        self.ball_expert = torch.nn.Sequential()
        self.ball_expert.add_module('ball_expert_layer_1_linear', nn.Linear(in_features=4 * 1, out_features=128 * 1))
        self.ball_expert.add_module('ball_expert_layer_2_linear', nn.Linear(128, 128))
        self.ball_expert.add_module('ball_expert_layer_3_linear', nn.Linear(128, 128))

        # self.ball_expert.add_module('ball_expert_layer_4_linear', nn.Linear(128, 64))
        # self.ball_expert.add_module('ball_expert_layer_5_linear', nn.Linear(64, 64))
        # self.ball_expert.add_module('ball_expert_layer_6_linear', nn.Linear(64, 64))

        # expert2
        self.blue_expert = torch.nn.Sequential()
        self.blue_expert.add_module('blue_expert_layer_1_linear', nn.Linear(in_features=21 * 1, out_features=128 * 1))
        self.blue_expert.add_module('blue_expert_layer_2_linear', nn.Linear(128, 128))
        self.blue_expert.add_module('blue_expert_layer_3_linear', nn.Linear(128, 128))

        # self.blue_expert.add_module('blue_expert_layer_4_linear', nn.Linear(128, 64))
        # self.blue_expert.add_module('blue_expert_layer_5_linear', nn.Linear(64, 64))
        # self.blue_expert.add_module('blue_expert_layer_6_linear', nn.Linear(64, 64))

        # expert3
        self.yellow_expert = torch.nn.Sequential()
        self.yellow_expert.add_module('yellow_expert_layer_1_linear',
                                      nn.Linear(in_features=15 * 1, out_features=128 * 1))
        self.yellow_expert.add_module('yellow_expert_layer_2_linear', nn.Linear(128, 128))
        self.yellow_expert.add_module('yellow_expert_layer_3_linear', nn.Linear(128, 128))

        # self.yellow_expert.add_module('yellow_expert_layer_4_linear', nn.Linear(128, 64))
        # self.yellow_expert.add_module('yellow_expert_layer_5_linear', nn.Linear(64, 64))
        # self.yellow_expert.add_module('yellow_expert_layer_6_linear', nn.Linear(64, 64))

        # gates
        # self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in range(num_task)]
        # for gate in self.gates:
        #     gate.data.normal_(0, 1)
        # self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]

        self.gate = torch.nn.Sequential()
        self.gate.add_module('gate_layer_1_liner', nn.Linear(in_features=40 * 1, out_features=3 * 1))

        # task tower
        # self.output_layers = torch.nn.Sequential()
        # self.output_layers.add_module('output_layer_1_linear', nn.Linear(128, 128))
        # self.output_layers.add_module('output_layer_2_batchnorm', nn.BatchNorm1d(128))
        # self.output_layers.add_module('output_layer_3_dropout', nn.Dropout(0.5))
        # self.output_layers.add_module('output_layer_last_linear', nn.Linear(128, output_size))

        self.output_layers = torch.nn.Sequential()
        self.output_layers.add_module('output_layer_1_linear', nn.Linear(128, 128))
        self.output_layers.add_module('output_layer_2_batchnorm', nn.BatchNorm1d(128))
        self.output_layers.add_module('output_layer_3_dropout', nn.Dropout(0.5))
        self.output_layers.add_module('output_layer_last_linear', nn.Linear(128, output_size))

    def forward(self, x):
        # assert x.size()[1] == len(self.feature_dict)
        # embedding
        # feature_list = list()
        # for feature, num in self.feature_dict.items():
        #     feature_list.append(x[:, num[1]].unsqueeze(1))

        # hidden layer
        # hidden = torch.cat(feature_list, axis=1).float() # batch * hidden_size

        # mmoe
        # experts_out = torch.einsum('ij, jkl -> ikl', x.float(), self.experts) # batch * mmoe_hidden_size * num_experts
        # experts_out += self.experts_bias
        # if self.expert_activation is not None:
        #     experts_out = self.expert_activation(experts_out)
        # print(experts_out.shape)[5, 128, 3]
        # print(x.float())#[5,40]
        # print(x[:,0:4].shape)

        expert1_out = torch.relu(self.ball_expert(x[:, 0:4].float()).unsqueeze(dim=2))  # [5,128,1]
        expert2_out = torch.relu(self.blue_expert(x[:, 4:25].float()).unsqueeze(dim=2))  # [5,128,1]
        expert3_out = torch.relu(self.yellow_expert(x[:, 25:40].float()).unsqueeze(dim=2))  # [5,128,1]

        # expert1_out = self.ball_expert(x[:,0:4].float()).unsqueeze(dim=2)#[5,128,1]
        # expert2_out = self.blue_expert(x[:,4:25].float()).unsqueeze(dim=2)#[5,128,1]
        # expert3_out = self.yellow_expert(x[:,25:40].float()).unsqueeze(dim=2)#[5,128,1]
        experts_out = torch.cat((expert1_out, expert2_out, expert3_out), dim=2)
        # print(experts_out.shape)

        # gates_out = list()
        # for idx, gate in enumerate(self.gates):
        #     gate_out = torch.einsum('ab, bc -> ac', x.float(), gate) # batch * num_experts
        #     if self.gates_bias:
        #         gate_out += self.gates_bias[idx]
        #     gate_out = nn.Softmax(dim=-1)(gate_out)
        #     print(gate_out)
        #     gates_out.append(gate_out)#weights for experts

        gate_out = nn.Softmax(dim=-1)(self.gate(x.float()))
        # print(gate_out)

        # outs = list()
        # for gate_output in gates_out:
        #     expanded_gate_output = torch.unsqueeze(gate_output, 1) # batch * 1 * num_experts
        #     weighted_expert_output = experts_out * expanded_gate_output.expand_as(experts_out) # batch * mmoe_hidden_size * num_experts
        #     #outs.append(torch.sum(weighted_expert_output, 2)) # batch * mmoe_hidden_size
        #     outs = torch.sum(weighted_expert_output, 2)

        expanded_gate_output = torch.unsqueeze(gate_out, 1)  # batch * 1 * num_experts
        weighted_expert_output = experts_out * expanded_gate_output.expand_as(
            experts_out)  # batch * mmoe_hidden_size * num_experts
        outs = torch.sum(weighted_expert_output, 2)

        # task tower
        # task_outputs = list()
        # for i in range(self.num_task):
        #     x = outs[i]
        #     for mod in getattr(self, 'task_{}_dnn'.format(i+1)):
        #         x = torch.floor(7 * torch.sigmoid(mod(x)))
        #     task_outputs.append(x)

        # task_outputs = list()
        # x = torch.floor(7 * torch.sigmoid(self.output_layers(outs)))
        # task_outputs = torch.softmax(self.output_layers(outs),dim=1)
        task_outputs = self.output_layers(outs)
        # x = torch.softmax(self.output_layers(outs),dim=1)
        # task_outputs.append(x)

        return task_outputs
