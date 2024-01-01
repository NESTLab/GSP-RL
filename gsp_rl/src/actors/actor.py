import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam

import numpy as np
import math

from gsp_rl.src.buffers import(
    ReplayBuffer,
    SequenceReplayBuffer,
    AttentionSequenceReplayBuffer
)
from gsp_rl.src.actors.learning_aids import NetworkAids

class Actor(NetworkAids):
    '''
    This class will be the foundation class for Agent and will hold all specific functions
    '''
    def __init__(self, id, n_obs, n_actions, options_per_action, n_agents, n_chars, meta_param_size, 
                 intention = False, recurrent_intention = False, attention = False, gnn=False, 
                 intention_neighbors = False, intention_look_back = 2, seq_len=5):

        super().__init__()

        self.id = id
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.options_per_action = options_per_action

        self.n_chars = n_chars
        self.n_agents = n_agents
        self.meta_param_size = meta_param_size

        self.action_space = [i for i in range(self.options_per_action**self.n_actions)]
        self.failure_action_code = len(self.action_space)

        self.intention = intention
        self.recurrent_intention = recurrent_intention
        self.attention_intention = attention
        self.gnn_intention = gnn
        self.intention_neighbors = intention_neighbors
        self.intention_look_back = intention_look_back
        self.seq_len = seq_len

        self.network_input_size = self.n_obs
        if self.intention:
            self.network_input_size += 1  
        if self.intention_neighbors:
            self.intention_network_input = 2+2*2  # [own prox, neighbors prox, own prev gsp, neighbors prev gsp]
            if self.attention_intention:
                self.attention_observation = [[0 for _ in range(self.intention_network_input)] for _ in range(self.seq_len)]
            #TODO write logic for recurrent and attention
        else:
            self.intention_network_input = self.n_agents*self.n_chars
            if self.attention_intention:  
                self.attention_observation = [[0 for _ in range(2+self.n_agents*self.n_chars)] for _ in range(self.seq_len)]
            elif self.recurrent_intention:
                self.recurrent_intention_network_input = self.intention_network_input + self.meta_param_size
            

    def build_networks(self, learning_scheme):
        if learning_scheme == 'None':
            self.networks = {'learning_scheme': '', 'learn_step_counter': 0}
        if learning_scheme == 'DQN':
            nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                   'num_ops_per_action':self.options_per_action}
            self.networks = self.build_DQN(nn_args)
            self.networks['learning_scheme'] = 'DQN'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete')
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'DDQN':
            nn_args = {'id':self.id, 'lr':self.lr, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                   'num_ops_per_action':self.options_per_action}
            self.networks = self.build_DDQN(nn_args)
            self.networks['learning_scheme'] = 'DDQN'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete')
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'DDPG':
            actor_nn_args = {'id':self.id, 'num_actions':self.n_actions, 'observation_size':self.network_input_size,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
            critic_nn_args = {'id':self.id, 'num_actions':self.n_actions, 'lr': self.lr, 'observation_size':self.network_input_size}
            self.networks = self.build_DDPG(actor_nn_args, critic_nn_args)
            self.networks['learning_scheme'] = 'DDPG'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.n_actions, 'Continuous')
            self.networks['n_actions'] = self.n_actions
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'TD3':
            actor_nn_args = {'id':self.id, 'alpha':self.alpha, 'input_dims': self.network_input_size, 'fc1_dims':400,
                         'fc2_dims':300, 'n_actions':self.n_actions}
            critic_nn_args = {'id':self.id, 'beta':self.beta, 'input_dims':self.network_input_size, 'fc1_dims':400,
                          'fc2_dims':300, 'n_actions':self.n_actions}
            self.networks = self.build_TD3(actor_nn_args, critic_nn_args)
            self.networks['learning_scheme'] = 'TD3'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.n_actions, 'Continuous')
            self.networks['n_actions'] = self.n_actions
            self.networks['learn_step_counter'] = 0
        else:
            print("removed the exception")
            #raise Exception('[ERROR] Learning scheme is not recognised: '+ learning_scheme)


    def build_DQN(self, nn_args):
        return self.make_DQN_networks(nn_args)
    
    def build_DDQN(self, nn_args):
        return self.make_DDQN_networks(nn_args)
    
    def build_DDPG(self, actor_nn_args, critic_nn_args):
        return self.make_DDPG_networks(actor_nn_args, critic_nn_args)

    def build_TD3(self, actor_nn_args, critic_nn_args):
        return self.make_TD3_networks(actor_nn_args, critic_nn_args)

    def build_intention_network(self, learning_scheme):
        if self.attention_intention:
            nn_args = {'embed_size':256, 'num_layers':8, 'heads':8, 'forward_expansion':4, 'dropout':0, 'max_length':self.seq_len}
            self.intention_networks = self.make_Attention_Encoder(nn_args)
            self.intention_networks['learning_scheme'] = 'attention'
            self.intention_networks['replay'] = AttentionSequenceReplayBuffer(max_sequence=100, num_observations = self.intention_network_input, seq_len = 5)
            self.intention_networks['learn_step_counter'] = 0
        else:
            if learning_scheme == 'DDPG':
                if self.recurrent_intention:
                    self.intention_networks = self.build_RDDPG_intention()
                    self.intention_networks['learning_scheme'] = 'RDDPG'
                    self.intention_networks['n_actions'] = 1
                    self.intention_networks['replay'] = SequenceReplayBuffer(max_sequence=100, num_observations = self.intention_network_input, num_actions = 1, seq_len = 5)
                    self.intention_networks['learn_step_counter'] = 0
                else:
                    self.intention_networks = self.build_DDPG_intention()
                    self.intention_networks['learning_scheme'] = 'DDPG'
                    self.intention_networks['n_actions'] = 1
                    self.intention_networks['replay'] = ReplayBuffer(self.mem_size, self.intention_network_input, 1, 'Continuous', use_intention = True)
                    self.intention_networks['learn_step_counter'] = 0
            elif learning_scheme == 'TD3':
                if self.recurrent_intention:
                    self.intention_networks = self.build_RTD3_intention()
                    self.intention_networks['learning_scheme'] = 'RTD3'
                    self.intention_networks['n_actions']  = 1
                    self.intention_networks['replay'] = SequenceReplayBuffer(max_sequence=100, num_observations = self.intention_network_input, num_actions = 1, seq_len = 5)
                    self.intention_networks['learn_step_counter'] = 0
                else:
                    self.intention_networks = self.build_TD3_intention()
                    self.intention_networks['learning_scheme'] = 'TD3'
                    self.intention_networks['n_actions'] = 1
                    self.intention_networks['replay'] = ReplayBuffer(self.mem_size, self.intention_network_input, 1, 'Continuous', use_intention = True)
                    self.intention_networks['learn_step_counter'] = 0
            else:
                raise Exception('[Error] Intention learning scheme is not recognised: '+learning_scheme)

    def build_DDPG_intention(self):
        actor_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':self.intention_network_input,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':1, 'lr': self.lr, 'observation_size':self.intention_network_input}
        return self.make_DDPG_networks(actor_nn_args, critic_nn_args)
    
    def build_RDDPG_intention(self):
        actor_nn_args = {'id':self.id, 'num_actions':1, 'observation_size':self.recurrent_intention_network_input,
                         'lr': self.lr, 'min_max_action':self.min_max_action}
        critic_nn_args = {'id':self.id, 'num_actions':1, 'lr': self.lr, 'observation_size':self.recurrent_intention_network_input}
        ee_nn_args = {'observation_size': 18, 'hidden_size':self.meta_param_size, 'meta_param_size': self.meta_param_size, 'batch_size': self.batch_size, 'num_layers':1, 'lr': self.ee_lr}
        Networks = self.make_DDPG_networks(actor_nn_args, critic_nn_args)
        Networks['ee'] = self.make_EE_networks(ee_nn_args)
        return Networks

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        # Update Intention Networks 
        if self.intention:
            if self.intention_networks['learning_scheme'] == 'DDPG' or self.intention_networks['learning_scheme'] == 'RDDPG':
                self.intention_networks = self.update_DDPG_network_parameters(tau, self.intention_networks)
            elif self.intention_networks['learning_scheme'] == 'TD3' or self.intention_networks['learning_scheme'] == 'RTD3':
                self.intention_networks = self.update_TD3_network_parameters(tau, self.intention_networks)
        # Update Action Selection Networks
        if self.networks['learning_scheme'] == 'DDPG':
            self.networks = self.update_DDPG_network_parameters(tau, self.networks)
        elif self.networks['learning_scheme'] == 'TD3':
            self.networks = self.update_TD3_network_parameters(tau, self.networks)

    def replace_target_network(self):
        if self.networks['learn_step_counter'] % self.replace_target_ctr==0:
            self.networks['q_next'].load_state_dict(self.networks['q_eval'].state_dict())
 
    def choose_action(self, observation, networks, test=False):        
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            if test or np.random.random()>self.epsilon:
                actions = self.DQN_DDQN_choose_action(observation, networks)
            else:
                actions = np.random.choice(self.action_space)
            return actions
        elif networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
            actions = self.DDPG_choose_action(observation, networks)
            if not test:
                actions+=T.normal(0.0, self.noise, size = (1, networks['n_actions'])).to(networks['actor'].device)
            actions = T.clamp(actions, -self.min_max_action, self.min_max_action)
            return actions[0].cpu().detach().numpy()
        elif networks['learning_scheme'] == 'TD3':
            return self.TD3_choose_action(observation, networks, self.n_actions)
        elif networks['learning_scheme'] == 'attention':
            self.attention_observation.append(observation)
            self.attention_observation.pop(0)
            #print(type(observation))
            observation = np.array(self.attention_observation)
            #print(type(observation))
            observation = T.Tensor(observation).to(networks['attention'].device)
            #print(type(observation))
            return self.Attention_choose_action(observation.unsqueeze(0), networks)
        else:
            raise Exception('[ERROR]: Learning scheme not recognised for action selection ' + networks['learning_scheme'])
    
    def learn(self, edge_index=None):
        if self.networks['replay'].mem_ctr < (self.n_agents*self.batch_size + self.batch_size):
                return

        if self.intention:
            if self.networks['learn_step_counter'] % self.intn_learning_offset == 0:
                #print('[DEBUG] Learning Attention', self.networks['learn_step_counter'])
                self.learn_intention(edge_index)

        if self.networks['learning_scheme'] == 'DQN':
            self.replace_target_network()
            
            return self.learn_DQN(self.networks)

        elif self.networks['learning_scheme'] == 'DDQN':
            self.replace_target_network()
            return self.learn_DDQN(self.networks)

        elif self.networks['learning_scheme'] == 'DDPG':
            self.update_network_parameters()
            return self.learn_DDPG(self.networks)

        elif self.networks['learning_scheme'] == 'TD3':
            self.update_network_parameters()
            return self.learn_TD3(self.networks)

    def learn_intention(self, edge_index=None):
        if self.intention_networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
            self.learn_DDPG(self.intention_networks, self.intention, self.recurrent_intention, edge_index)
        elif self.intention_networks['learning_scheme'] == 'TD3':
            self.learn_TD3(self.intention_networks, self.intention, self.recurrent_intention)
        elif self.intention_networks['learning_scheme'] == 'attention':
            self.learn_attention(self.intention_networks)

    def store_agent_transition(self, s, a, r, s_, d):
        self.store_transition(s, a, r, s_, d, self.networks)
    
    def store_intention_transition(self, s, a, r, s_, d):
        if self.attention_intention:
            self.store_attention_transition(s, a, self.intention_networks)
        else:
            self.store_transition(s, a, r, s_, d, self.intention_networks)

    def reset_intention_sequence(self):
        self.intention_sequence = [np.zeros(self.intention_network_input) for i in range(self.seq_len)]
    
    def add_intention_sequence(self, obs):
        self.intention_sequence.append(obs)
        self.intention_sequence.pop(0)

    def save_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].save_checkpoint(path)

        elif self.networks['learning_scheme'] == 'DDPG':
            self.networks['actor'].save_checkpoint(path)
            self.networks['target_actor'].save_checkpoint(path)
            self.networks['critic'].save_checkpoint(path)
            self.networks['target_critic'].save_checkpoint(path)

        elif self.networks['learning_scheme'] == 'TD3':
            self.networks['actor'].save_checkpoint(path)
            self.networks['target_actor'].save_checkpoint(path)
            self.networks['critic_1'].save_checkpoint(path)
            self.networks['target_critic_1'].save_checkpoint(path)
            self.networks['critic_2'].save_checkpoint(path)
            self.networks['target_critic_2'].save_checkpoint(path)
        if self.attention_intention:
            if self.intention_networks['learning_scheme'] == 'attention':
                self.intention_networks['attention'].save_checkpoint(path)
        elif self.intention:
            self.intention_networks['actor'].save_checkpoint(path, self.intention)
            self.intention_networks['target_actor'].save_checkpoint(path, self.intention)
            if self.intention_networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
                self.intention_networks['critic'].save_checkpoint(path, self.intention)
                self.intention_networks['target_critic'].save_checkpoint(path, self.intention)
            elif self.intention_networks['learning_scheme'] in {'TD3', 'RTD3'}:
                self.intention_networks['critic_1'].save_checkpoint(path, self.intention)
                self.intention_networks['target_critic_1'].save_checkpoint(path, self.intention)
                self.intention_networks['critic_2'].save_checkpoint(path, self.intention)
                self.intention_networks['target_critic_2'].save_checkpoint(path, self.intention)
            if self.recurrent_intention:
                self.intention_networks['ee'].save_checkpoint(path)

    def load_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].load_checkpoint(path)
            #print('-------------------- Weights ------------------')
            #for param in self.q_eval.parameters():
            #    print(param.data)
        elif self.networks['learning_scheme'] == 'DDPG':
            self.networks['actor'].load_checkpoint(path)
            self.networks['target_actor'].load_checkpoint(path)
            self.networks['critic'].load_checkpoint(path)
            self.networks['target_critic'].load_checkpoint(path)

        elif self.networks['learning_scheme'] == 'TD3':
            self.networks['actor'].load_checkpoint(path)
            self.networks['target_actor'].load_checkpoint(path)
            self.networks['critic_1'].load_checkpoint(path)
            self.networks['target_critic_1'].load_checkpoint(path)
            self.networks['critic_2'].load_checkpoint(path)
            self.networks['target_critic_2'].load_checkpoint(path)
        
        if self.attention_intention:
            if self.intention_networks['learning_scheme'] == 'attention':
                self.intention_networks['attention'].load_checkpoint(path)
        elif self.intention:
            self.intention_networks['actor'].load_checkpoint(path, self.intention)
            self.intention_networks['target_actor'].load_checkpoint(path, self.intention)
            if self.intention_networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
                self.intention_networks['critic'].load_checkpoint(path, self.intention)
                self.intention_networks['target_critic'].load_checkpoint(path, self.intention)
            elif self.intention_networks['learning_scheme'] in {'TD3', 'RTD3'}:
                self.intention_networks['critic_1'].load_checkpoint(path, self.intention)
                self.intention_networks['target_critic_1'].load_checkpoint(path, self.intention)
                self.intention_networks['critic_2'].load_checkpoint(path, self.intention)
                self.intention_networks['target_critic_2'].load_checkpoint(path, self.intention)
            if self.recurrent_intention:
                self.intention_networks['ee'].load_checkpoint(path)
        
    
if __name__=='__main__':
    agent_args = {'id':1, 'n_obs':32, 'n_actions':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'intention':False, 'recurrent_intention':False, 'intention_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['n_obs'])

    
    print('[TESTING] DQN')
    agent.build_networks('DQN')
    agent.networks['learn_step_counter'] = agent.replace_target_ctr
    agent.replace_target_network()
    observation = np.random.random(size = agent_args['n_obs'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['n_obs'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())

    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])

    print('[TESTING] DDQN')
    agent_args = {'id':1, 'n_obs':32, 'n_actions':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'intention':False, 'recurrent_intention':False, 'intention_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['n_obs'])
    agent.build_networks('DDQN')
    agent.networks['learn_step_counter'] = agent.replace_target_ctr
    agent.replace_target_network()
    observation = np.random.random(size = agent_args['n_obs'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['n_obs'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())
    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])
    
    print('[TESTING] DDPG and param update')
    agent_args = {'id':1, 'n_obs':32, 'n_actions':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'intention':False, 'recurrent_intention':False, 'intention_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['n_obs'])
    agent.build_networks('DDPG')
    agent.update_network_parameters()
    observation = np.random.random(size = agent_args['n_obs'])
    done = False
    for i in range(200):
        action = [None, agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['n_obs'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())

    print('[TESTING] TD3')
    agent_args = {'id':1, 'n_obs':32, 'n_actions':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'intention':False, 'recurrent_intention':False, 'intention_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['n_obs'])
    agent.build_networks('TD3')
    agent.update_network_parameters()
    observation = np.random.random(size = agent_args['n_obs'])
    done = False
    for i in range(200):
        action = [None, agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['n_obs'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())
    print(agent.networks['actor'])
    print(agent.networks['critic_1'])
    print(agent.networks['critic_2'])
    

    print('[TESTING] Intention DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, intention=True, recurrent_intention = False)
    agent.build_networks('DDPG')
    agent.build_intention_network('DDPG')
    agent.update_network_parameters()
    print(agent.intention_networks['actor'])
    print(agent.intention_networks['critic'])
    

    print('[TESTING] Recurrent Intention DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, intention=True, recurrent_intention = True)
    agent.build_networks('DDPG')
    agent.build_intention_network('DDPG')
    agent.update_network_parameters()
    print(agent.intention_networks['actor'])
    print(agent.intention_networks['critic'])
    print(agent.intention_networks['ee'])