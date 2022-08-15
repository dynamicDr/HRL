import copy
import math
import os
import pickle
import random
from typing import Dict

import gym
import numpy as np
import torch

from matd3 import MATD3
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv


class VSSMAEnv(VSSBaseEnv):
    """This environment controls N robots in a VSS soccer League 3v3 match


        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized
            0               Ball X
            1               Ball Y
            2               Ball Vx
            3               Ball Vy
            4 + (7 * i)     id i Blue Robot X
            5 + (7 * i)     id i Blue Robot Y
            6 + (7 * i)     id i Blue Robot sin(theta)
            7 + (7 * i)     id i Blue Robot cos(theta)
            8 + (7 * i)     id i Blue Robot Vx
            9  + (7 * i)    id i Blue Robot Vy
            10 + (7 * i)    id i Blue Robot v_theta
            25 + (5 * i)    id i Yellow Robot X
            26 + (5 * i)    id i Yellow Robot Y
            27 + (5 * i)    id i Yellow Robot Vx
            28 + (5 * i)    id i Yellow Robot Vy
            29 + (5 * i)    id i Yellow Robot v_theta
            -3              robot goal x
            -2              robot goal y
            -1              robot role 0=attacker 1=defender
        Actions:
            Type: Box(N, 2)
            For each blue robot in control:
                Num     Action
                0       Left Wheel Speed  (%)
                1       Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                For all robots:
                    Goal
                    Ball Potential Gradient
                Individual:
                    Move to Ball
                    Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self, n_robots_control=3):
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3,
                         time_step=0.025)

        self.first_step = True
        self.n_robots_control = n_robots_control
        self.action_space = gym.spaces.Box(low=-1,
                                           high=1,
                                           shape=(n_robots_control, 2))
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_robots_control, 43),
                                                dtype=np.float32)

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.individual_reward = {}
        self.v_wheel_deadzone = 0.05
        self.observation = None
        self.ou_actions = []
        self.goal = [[0, 0], [0, 0]]
        self.attacker = 0
        self.defender_1 = 1
        self.defender_2 = 2
        self.writer=None
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )


        print('Environment initialized')

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None
        self.individual_reward = {}
        for ou in self.ou_actions:
            ou.reset()
        self.first_step = True
        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        if self.first_step:
            self.first_step = False
        return observation, reward, done, self.reward_shaping_total

    def set_attacker_and_goal(self, goal):
        self.goal = goal
        self.attacker = self._get_closet_robot_idx([self.frame.ball.x, self.frame.ball.y])
        self.defender_1 = self._get_closet_robot_idx(goal[0], except_idx=self.attacker)
        for i in range(self.n_robots_control):
            if i != self.attacker and i != self.defender_1:
                self.defender_2 = i
        self._frame_to_observations()
        # print(f"{self.attacker},{self.defender_1},{self.defender_2}")

    def get_rotated_obs(self):
        robots_dict = dict()
        for i in range(self.n_robots_blue):
            robots_dict[i] = list()
            robots_dict[i].append(self.norm_pos(self.frame.robots_blue[i].x))
            robots_dict[i].append(self.norm_pos(self.frame.robots_blue[i].y))
            robots_dict[i].append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robots_dict[i].append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robots_dict[i].append(self.norm_v(self.frame.robots_blue[i].v_x))
            robots_dict[i].append(self.norm_v(self.frame.robots_blue[i].v_y))
            robots_dict[i].append(self.norm_w(self.frame.robots_blue[i].v_theta))

        rotaded_obs = list()
        for i in range(self.n_robots_control):
            aux_dict = {}
            aux_dict.update(robots_dict)
            rotated = list()
            rotated = rotated + aux_dict.pop(i)
            teammates = list(aux_dict.values())
            for teammate in teammates:
                rotated = rotated + teammate
            rotaded_obs.append(rotated)

        return rotaded_obs

    def _frame_to_observations(self):

        observations = list()
        robots = self.get_rotated_obs()
        for idx in range(self.n_robots_control):
            observation = []

            observation.append(self.norm_pos(self.frame.ball.x))
            observation.append(self.norm_pos(self.frame.ball.y))
            observation.append(self.norm_v(self.frame.ball.v_x))
            observation.append(self.norm_v(self.frame.ball.v_y))

            observation += robots[idx]
            if self.n_robots_yellow != 0:
                for i in range(self.n_robots_yellow):
                    observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
                    observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
                    observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
                    observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
                    observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

            # append goal and role
            if idx == self.attacker:
                observation.append(self.norm_pos(self.frame.ball.x))
                observation.append(self.norm_pos(self.frame.ball.y))
                observation.append(0)
            elif idx == self.defender_1:
                observation.append(self.norm_pos(self.goal[0][0]))
                observation.append(self.norm_pos(self.goal[0][1]))
                observation.append(1)
            elif idx == self.defender_2:
                observation.append(self.norm_pos(self.goal[1][0]))
                observation.append(self.norm_pos(self.goal[1][1]))
                observation.append(1)
            else:
                raise Exception(f"idx{idx} is neither attacker nor defender")
            observations.append(np.array(observation, dtype=np.float32))
        # Append coach observation
        observations.append(np.array(observations[0][:40], dtype=np.float32))
        observations = np.array(observations)
        self.observation = observations
        return observations

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        # Send random commands to the other robots
        for i in range(self.n_robots_control):
            self.actions[i] = actions[i]
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[i])
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        for i in range(self.n_robots_control, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0])
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        for i in range(self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue + i].sample()

            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0])
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        return commands

    def _calculate_reward_and_done(self):
        reward = {f'robot_{i}': 0 for i in range(self.n_robots_control)}
        done = False
        # for agent
        w_move = 0.2  # [-5,5]
        w_ball_grad = 0.8  # [-5,5]
        w_energy = 2e-6
        w_speed = 1  # 0 or -1
        w_goal = 100

        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'goal_score': 0, 'ball_grad': 0,
                                         'goals_blue': 0, 'goals_yellow': 0}
        if len(self.individual_reward) == 0:
            for i in range(self.n_robots_control):
                self.individual_reward[f'robot_{i}'] = {'move': 0, 'energy': 0, 'speed': 0}

        # Check if goal
        if self.frame.ball.x > (self.field.length / 2):
            self.reward_shaping_total['goal_score'] += 1
            self.reward_shaping_total['goals_blue'] += 1
            for i in range(self.n_robots_control):
                reward[f'robot_{i}'] = w_goal * 1
            done = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.reward_shaping_total['goal_score'] -= 1
            self.reward_shaping_total['goals_yellow'] += 1
            for i in range(self.n_robots_control):
                reward[f'robot_{i}'] = w_goal * -1
            done = True
        else:
            # if not goal
            if self.last_frame is not None:
                grad_ball_potential, closest_move, move_reward, energy_penalty, speed_penalty = 0, 0, 0, 0, 0
                # Calculate ball potential
                grad_ball_potential = self._ball_grad()
                self.reward_shaping_total['ball_grad'] = w_ball_grad * grad_ball_potential  # noqa
                dead_robot_count = 0
                for idx in range(self.n_robots_control):
                    # Calculate Move reward
                    if w_move != 0:
                        if idx == self.attacker:
                            move_target = [self.frame.ball.x, self.frame.ball.y]
                        elif idx == self.defender_1:
                            move_target = self.goal[0]
                        elif idx == self.defender_2:
                            move_target = self.goal[1]
                        else:
                            raise Exception(f"idx{idx} is neither attacker nor defender")
                        move_reward = self._move_reward(idx, move_target)

                    # Calculate Energy penalty
                    if w_energy != 0:
                        energy_penalty = self._energy_penalty(robot_idx=idx)

                    # Calculate speed penalty
                    if w_speed != 0:
                        speed_dead_zone = 0.1
                        speed_x = self.observation[0][8 + 7 * idx]
                        speed_y = self.observation[0][9 + 7 * idx]
                        speed_abs = math.sqrt(math.pow(speed_x, 2) + math.pow(speed_y, 2))
                        speed_penalty = 0
                        if speed_abs <= speed_dead_zone:
                            speed_penalty = -1

                    rew = w_ball_grad * grad_ball_potential + \
                          w_move * move_reward + \
                          w_energy * energy_penalty + \
                          w_speed * speed_penalty

                    if speed_penalty < 0:
                        dead_robot_count += 1

                    reward[f'robot_{idx}'] = rew
                    self.individual_reward[f'robot_{idx}']['move'] = w_move * move_reward  # noqa
                    self.individual_reward[f'robot_{idx}']['energy'] = w_energy * energy_penalty  # noqa
                    self.individual_reward[f'robot_{idx}']['speed'] = w_speed * speed_penalty  # noqa
        return reward, done

    def write_log(self,writer,step_num):
        if self.writer is None:
            self.writer = writer
        self.writer.add_scalar(f'Ball Grad Reward', self.reward_shaping_total['ball_grad'], global_step=step_num)
        for idx in range(self.n_robots_control):
            self.writer.add_scalar(f'Agent_{idx} Move Reward', self.individual_reward[f'robot_{idx}']['move'], global_step=step_num)
            self.writer.add_scalar(f'Agent_{idx} Energy Penalty', self.individual_reward[f'robot_{idx}']['energy'], global_step=step_num)
            self.writer.add_scalar(f'Agent_{idx} Speed Penalty', self.individual_reward[f'robot_{idx}']['speed'], global_step=step_num)


    def _get_closet_robot_idx(self, target, except_idx=None):
        robots_distance_to_target = {}
        for idx in range(self.n_robots_control):
            target = np.array(target)
            robot = np.array([self.frame.robots_blue[idx].x,
                              self.frame.robots_blue[idx].y])
            robot_distance_to_target = np.sqrt(sum((robot - target) ** 2 for robot, target in zip(robot, target)))
            robots_distance_to_target[idx] = robot_distance_to_target
        sorted_list = sorted(robots_distance_to_target.items(), key=lambda kv: [kv[1], kv[0]])
        if except_idx is not None:
            if sorted_list[0][0] == except_idx:
                return sorted_list[1][0]
        return sorted_list[0][0]

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x():
            return random.uniform(-field_half_length + 0.1,
                                  field_half_length - 0.1)

        def y():
            return random.uniform(-field_half_width + 0.1,
                                  field_half_width - 0.1)

        def theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed, right_wheel_speed

    def _ball_grad(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0) \
                      + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step,
                                          -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def _move_reward(self, robot_idx, target):
        '''Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''

        target = np.array(target)
        robot = np.array([self.frame.robots_blue[robot_idx].x,
                          self.frame.robots_blue[robot_idx].y])
        robot_vel = np.array([self.frame.robots_blue[robot_idx].v_x,
                              self.frame.robots_blue[robot_idx].v_y])
        robot_ball = target - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def _energy_penalty(self, robot_idx: int):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[robot_idx].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[robot_idx].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty


class VSSMAOpp(VSSMAEnv):

    def __init__(self, n_robots_control=3):
        super().__init__(n_robots_control=n_robots_control)
        self.args = None
        self.opps = []
        self.load_opp()

    def load_opp(self):
        self.opps = []
        try:
            with open(os.path.dirname(os.path.realpath(__file__)) + f'/opponent/args.pkl', 'rb') as f:
                self.args = pickle.load(f)
            device = torch.device('cpu')
            for i in range(self.n_robots_yellow):
                ckp_path = os.path.dirname(os.path.realpath(__file__)) \
                           + f'/opponent/opp_{i}.pth'
                agent = MATD3(self.args, i, None)
                state_dict = torch.load(ckp_path)
                agent.actor.load_state_dict(state_dict)
                agent.actor.eval()
                self.opps.append(agent)
                print(f"Successfully load opponents. ckp_path:{ckp_path}")
        except FileNotFoundError:
            print("No ckp found. Will use random opponents.")

    def _opp_obs(self):
        observation = []
        observation.append(self.norm_pos(-self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(-self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        #  we reflect the side that the opp is attacking,
        #  so that he will attack towards the goal where the goalkeeper is
        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(-self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

            observation.append(
                np.sin(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(
                -np.cos(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(self.norm_v(-self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))

            observation.append(self.norm_w(-self.frame.robots_yellow[i].v_theta))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(-self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(self.norm_v(-self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(-self.frame.robots_blue[i].v_theta))

        # get rotated
        observations = []
        observations.append(observation)
        player_0 = copy.deepcopy(observation[4 + (7 * 0): 11 + (7 * 0)])
        player_1 = copy.deepcopy(observation[4 + (7 * 1): 11 + (7 * 1)])
        player_2 = copy.deepcopy(observation[4 + (7 * 2): 11 + (7 * 2)])
        observation_1 = copy.deepcopy(observation)
        observation_1[4 + (7 * 0): 11 + (7 * 0)] = player_1
        observation_1[4 + (7 * 1): 11 + (7 * 1)] = player_0
        observations.append(observation_1)
        observation_2 = copy.deepcopy(observation)
        observation_2[4 + (7 * 0): 11 + (7 * 0)] = player_2
        observation_2[4 + (7 * 2): 11 + (7 * 2)] = player_0
        observations.append(observation_2)
        return np.array(observations, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        for i in range(self.n_robots_blue):
            self.actions[i] = actions[i]
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[i])
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        opp_obs = self._opp_obs()

        for i in range(self.n_robots_yellow):
            if len(self.opps) != 0:
                a = self.opps[i].choose_action(opp_obs[i], noise_std=0)
                opp_action = copy.deepcopy(a)
                # print(opp_action)
            else:
                # print("ou_action")
                opp_action = self.ou_actions[self.n_robots_blue + i].sample()[i]
            v_wheel1, v_wheel0 = self._actions_to_v_wheels(opp_action)
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        return commands


class VSSMASelfplay(VSSMAEnv):

    def __init__(self, n_robots_control=3):
        super().__init__(n_robots_control=n_robots_control)
        self.args = None
        self.opps = []
        self.load_opp()

    def load_opp(self):
        self.opps = []
        try:
            with open(os.path.dirname(os.path.realpath(__file__)) + f'/opponent/args.pkl', 'rb') as f:
                self.args = pickle.load(f)
            device = torch.device('cpu')
            for i in range(self.n_robots_yellow):
                ckp_path = os.path.dirname(os.path.realpath(__file__)) \
                           + f'/opponent/opp_{i}.pth'
                agent = MATD3(self.args, i, None)
                state_dict = torch.load(ckp_path)
                agent.actor.load_state_dict(state_dict)
                agent.actor.eval()
                self.opps.append(agent)
                print(f"Successfully load opponents. ckp_path:{ckp_path}")
        except FileNotFoundError:
            print("No ckp found. Will use random opponents.")

    def _opp_obs(self):
        observation = []
        observation.append(self.norm_pos(-self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(-self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        #  we reflect the side that the opp is attacking,
        #  so that he will attack towards the goal where the goalkeeper is
        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(-self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

            observation.append(
                np.sin(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(
                -np.cos(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(self.norm_v(-self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))

            observation.append(self.norm_w(-self.frame.robots_yellow[i].v_theta))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(-self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(self.norm_v(-self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(-self.frame.robots_blue[i].v_theta))

        # get rotated
        observations = []
        observations.append(observation)
        player_0 = copy.deepcopy(observation[4 + (7 * 0): 11 + (7 * 0)])
        player_1 = copy.deepcopy(observation[4 + (7 * 1): 11 + (7 * 1)])
        player_2 = copy.deepcopy(observation[4 + (7 * 2): 11 + (7 * 2)])
        observation_1 = copy.deepcopy(observation)
        observation_1[4 + (7 * 0): 11 + (7 * 0)] = player_1
        observation_1[4 + (7 * 1): 11 + (7 * 1)] = player_0
        observations.append(observation_1)
        observation_2 = copy.deepcopy(observation)
        observation_2[4 + (7 * 0): 11 + (7 * 0)] = player_2
        observation_2[4 + (7 * 2): 11 + (7 * 2)] = player_0
        observations.append(observation_2)
        return np.array(observations, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        for i in range(self.n_robots_blue):
            self.actions[i] = actions[i]
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[i])
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        opp_obs = self._opp_obs()

        for i in range(self.n_robots_yellow):
            if len(self.opps) != 0:
                a = self.opps[i].choose_action(opp_obs[i], noise_std=0)
                opp_action = copy.deepcopy(a)
                # print(opp_action)
            else:
                # print("ou_action")
                opp_action = self.ou_actions[self.n_robots_blue + i].sample()[i]
            v_wheel1, v_wheel0 = self._actions_to_v_wheels(opp_action)
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        return commands


if __name__ == '__main__':
    env = VSSMAEnv()
    print(env.n_robots_yellow)