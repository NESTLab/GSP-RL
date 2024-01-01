from gsp_rl.src.actors import Actor

import numpy as np
import math
import torch.nn as nn

Loss = nn.MSELoss()

class Agent(Actor):
    def __init__(self, n_agents, n_obs, n_actions, options_per_action, id, learning_scheme,
                 n_chars=4, intention_look_back = 2, min_max_action = 1, use_intention=False, 
                 use_recurrent=False, attention=False, gnn=False, intention_neighbors=False,
                 meta_param_size = 1, seq_len=5, edge_index = None, prox_filter_angle = 45):

        args = {'id':id, 'n_obs':n_obs, 'n_actions':n_actions, 'options_per_action':options_per_action, 'n_agents':n_agents,
                'n_chars':n_chars, 'meta_param_size':meta_param_size, 'intention':use_intention, 'recurrent_intention':use_recurrent,
                'attention':attention, 'gnn':gnn, 'intention_neighbors':intention_neighbors, 'intention_look_back':intention_look_back,
                'seq_len':seq_len}

        super().__init__(**args)
        self.n_agents = n_agents
        self.learning_scheme = learning_scheme
        self.object_stats = []
        self.neighbors = {}
        self.min_obj_stats = np.zeros(4) # vel, accel, ang_vel, ang_accel
        self.max_obj_stats = np.zeros(4)
        self.decimals = 2
        min_max = 1.25
        bins = 8
        self.angle_bins = np.arange(-180, 180, 360/bins)
        self.acceleration_bins = np.around(np.arange(-min_max, min_max, (min_max*2)/bins), self.decimals)
        self.binned_angle = None
        self.binned_acceleration = None
        self.obj_state = None
        self.edge_index = edge_index
        
        self.ROBOT_PROXIMITY_ANGLES = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5,
                                       112.5, 127.5, 142.5, 157.5, 172.5, -172.5, 
                                       -157.5, -142.5, -127.5, -112.5, -97.5, 
                                       -82.5, -67.5, -52.5, -37.5, -22.5, -7.5]
        self.prox_filter_angle = prox_filter_angle
        
        self.build_networks(learning_scheme)

        if self.intention:
            self.build_intention_network('DDPG')
            if self.intention_neighbors:
                self.build_neighbors()

    def make_agent_state(self, env_obs, heading_intention=None, global_knowledge=None):
        if heading_intention is not None:
            if global_knowledge is not None:
                env_obs = np.concatenate((env_obs, np.array([heading_intention]), global_knowledge)) 
            else:
                env_obs = np.concatenate((env_obs, np.array([heading_intention]))) 
        elif global_knowledge is not None:
            env_obs = np.concatenate((env_obs, global_knowledge))
        return env_obs   

    def angle_difference(self, a1, a2):
        diff = a1 - a2
        while diff < -180.0:
            diff += 360.0
        while diff > 180.0:
            diff -= 360
        return diff

    def store_object_stats(self, obj_stats, calculate):
        # len = 3 is so that we can calculate velocity and acceleration
        # calculate is a flag that tracks based on episode time steps.
        # we dont want to do any calculations within the first 3 time
        # steps of an episode.
        if not calculate:
            self.object_stats.append(obj_stats)
        else:
            # get rid of the oldest stat
            self.object_stats.pop(0)
            self.object_stats.append(obj_stats)
            velocity_t0 = math.sqrt((self.object_stats[2][0] - self.object_stats[1][0])**2 + (self.object_stats[2][1] - self.object_stats[1][1])**2)/0.1
            velocity_t1 = math.sqrt((self.object_stats[1][0] - self.object_stats[0][0])**2 + (self.object_stats[1][1] - self.object_stats[0][1])**2)/0.1
            acceleration = (velocity_t0 - velocity_t1)/0.1

            angular_velocity_t0 = self.angle_difference(self.object_stats[2][5], self.object_stats[1][5])/0.1

            angular_velocity_t1 = self.angle_difference(self.object_stats[1][5], self.object_stats[0][5])/0.1
            angular_acceleration = (angular_velocity_t0 - angular_velocity_t1)/0.1

            self.binned_acceleration = int(min(self.acceleration_bins, key=lambda x:abs(x-acceleration))*(10**self.decimals))
            self.binned_angle = int(min(self.angle_bins, key=lambda x:abs(x-self.object_stats[2][5])))
            self.obj_state = [self.binned_acceleration, self.binned_angle]


            # This is for finding bin limits only
            if velocity_t0 > self.max_obj_stats[0]: self.max_obj_stats[0] = velocity_t0
            if velocity_t0 < self.min_obj_stats[0]: self.min_obj_stats[0] = velocity_t0
            if acceleration > self.max_obj_stats[1]: self.max_obj_stats[1] = acceleration
            if acceleration < self.min_obj_stats[1]: self.min_obj_stats[1] = acceleration
            if angular_velocity_t0 > self.max_obj_stats[2]: self.max_obj_stats[2] = angular_velocity_t0
            if angular_velocity_t0 < self.min_obj_stats[2]: self.min_obj_stats[2] = angular_velocity_t0
            if angular_acceleration > self.max_obj_stats[3]: self.max_obj_stats[3] = angular_acceleration
            if angular_acceleration < self.min_obj_stats[3]: self.min_obj_stats[3] = angular_acceleration

    def reset_obj_stats(self):
        self.object_stats = []        

    def choose_agent_action(self, observation, failures, test=False):
        if self.learning_scheme == 'None':
            # Not sure what to do here for no learning
            return [0, 0, 0], 0

        if failures:
            self.failed = True
            return self.failure_action, self.failure_action_code

        self.failed = False
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            action_num = self.choose_action(observation, self.networks, test)
            actions = self.parse_action(action_num)

        if self.networks['learning_scheme'] == 'DDPG' or self.networks['learning_scheme'] == 'TD3':
            actions = self.choose_action(observation, self.networks, test)
            actions = np.pad(actions, (0, 1))
            action_num = None

        return actions, action_num

    
    def parse_action(self, action_num):
        '''
        This function will parse the number action to
        a set of wheel actions:

        0 - (- 1,-1)
        1 - (-1, 0)
        2 - (-1, 1)
        3 - (0, -1)
        4 - (0, 0)
        5 - (0, 1)
        6 - (1, -1)
        7 - (1, 0)
        8 - (1, 1)
        '''
        if action_num < 0 or action_num >=self.options_per_action**self.n_actions:
            raise Exception('Action Number Out of Range:'+str(action_num))
        l_wheel = round((math.floor(action_num/self.options_per_action) - 1)/10.0, 1)
        r_wheel = round((action_num%self.options_per_action - 1)/10.0, 1)
        # Trailing zero is hardcoded control for gripper
        return np.array([l_wheel, r_wheel, 0])
        

    def choose_object_intention(self, agent_intention_states, edge_index = None, test = False):
        if self.intention_neighbors:
            return [self.choose_action(agent_intention_states[i], self.intention_networks, edge_index, test) for i in range(len(agent_intention_states))]
        else:
            observation = np.array(agent_intention_states)
            if edge_index is not None:
                observation = observation.reshape(1, observation.shape[0]).repeat(self.n_agents, axis = 0)
            return self.choose_action(observation, self.intention_networks, edge_index, test) 

    def filter_prox_values(self, prox_values, angle_to_cyl):
        if angle_to_cyl > 0:
            if angle_to_cyl > 180-self.prox_filter_angle:
                cw_lim = angle_to_cyl + self.prox_filter_angle - 360
            else:
                cw_lim = angle_to_cyl+self.prox_filter_angle
            ccw_lim = angle_to_cyl - self.prox_filter_angle
        elif angle_to_cyl < 0:
            if angle_to_cyl < -180 +self.prox_filter_angle:
                ccw_lim = angle_to_cyl-self.prox_filter_angle+360
            else:
                ccw_lim = angle_to_cyl - self.prox_filter_angle
            cw_lim = angle_to_cyl + self.prox_filter_angle
        else:
            cw_lim = self.prox_filter_angle
            ccw_lim = -self.prox_filter_angle

        index = []
        filtered_prox_values = []
        if angle_to_cyl > 180 - self.prox_filter_angle:
            for i in range(len(self.ROBOT_PROXIMITY_ANGLES)):
                if self.ROBOT_PROXIMITY_ANGLES[i] > ccw_lim:
                    index.append(i)
                elif self.ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i])
        elif angle_to_cyl < -180+self.prox_filter_angle:
            for i in range(len(self.ROBOT_PROXIMITY_ANGLES)):
                if self.ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                elif self.ROBOT_PROXIMITY_ANGLES[i] > ccw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i]) 
        else:
            for i in range(len(self.ROBOT_PROXIMITY_ANGLES)):
                if self.ROBOT_PROXIMITY_ANGLES[i] > ccw_lim and self.ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i])
        return filtered_prox_values, index

    def build_neighbors(self):
        agents_available = np.arange(self.n_agents)
        for agent in range(self.n_agents):
            self.neighbors[agent] = [agents_available[agent-1], agents_available[(agent+1)%self.n_agents]]
    
    def build_intention_states(self, agent_prox_values, agent_prev_intention):
        states = []
        for agent in self.neighbors.keys():
            agent_state = np.zeros(self.intention_network_input)
            n1, n2 = self.neighbors[agent]
            agent_state[0] = agent_prox_values[agent]
            agent_state[1] = agent_prox_values[n1]
            agent_state[2] = agent_prox_values[n2]
            agent_state[3] = agent_prev_intention[agent]
            agent_state[4] = agent_prev_intention[n1]
            agent_state[5] = agent_prev_intention[n2]
            states.append(agent_state)
        return states