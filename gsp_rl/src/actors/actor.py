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
    def __init__(
            self,
            id: int,
            config: dict,
            network: str,
            input_size: int,
            output_size: int,  
            min_max_action: int,
            meta_param_size: int,
            gsp: bool = False,
            recurrent_gsp: bool = False,
            attention: bool = False, 
            recurrent_hidden_size: int = 256,
            recurrent_embedding_size: int = 256,
            recurrent_num_layers = 5,
            gsp_input_size: int = 6,
            gsp_output_size: int = 1,
            gsp_min_max_action: float = 1.0,
            gsp_look_back: int = 2,
            gsp_sequence_length: int = 5
        ) -> None:
        """
        id: int -> the id of the agent
        input_size: int -> the size of the observation space coming from the environment
        output_size: int -> the size of the expected action space
        meta_param_size: int -> the encoding size for LSTM
        gsp: bool -> flag to use DDPG-GSP
        recurrent_gsp: bool -> flag to use RDDPG-GSP
        attention: bool -> flag to use A-GSP
        gsp_input_size: int -> the input size to the gsp network
        gsp_output_size: int -> the output size of the gsp network
        gsp_look_back: int -> ...
        seq_len: int -> length of sequence to use as input to A-GSP
        """
        super().__init__(config)

        self.id = id
        self.input_size = input_size
        self.output_size = output_size

        self.min_max_action = min_max_action
        self.meta_param_size = meta_param_size

        self.action_space = [i for i in range(self.output_size)]
        self.failure_action_code = len(self.action_space)

        self.gsp = gsp
        self.recurrent_gsp = recurrent_gsp
        self.attention_gsp = attention
        
        self.gsp_network_input = gsp_input_size
        self.gsp_network_output = gsp_output_size
        self.gsp_min_max_action = gsp_min_max_action
        self.gsp_look_back = gsp_look_back
        self.gsp_sequence_length = gsp_sequence_length

        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_embedding_size = recurrent_embedding_size
        self.recurrent_num_layers = recurrent_num_layers

        self.network_input_size = self.input_size
        if self.gsp:
            self.network_input_size += self.gsp_network_output 
        if self.attention_gsp:  
            self.attention_observation = [[0 for _ in range(self.gsp_network_input)] for _ in range(self.gsp_sequence_length)]
        elif self.recurrent_gsp:
            self.recurrent_gsp_network_input = self.gsp_network_input

        self.build_networks(network)
        if gsp is not None:
            if attention:
                self.build_gsp_network('attention')
            self.build_gsp_network('DDPG')
            
    def build_networks(self, learning_scheme):
        if learning_scheme == 'None':
            self.networks = {'learning_scheme': '', 'learn_step_counter': 0}
        if learning_scheme == 'DQN':
            nn_args = {
                'id':self.id,
                'lr':self.lr,
                'output_size':self.output_size,
                'input_size':self.network_input_size,
            }
            self.networks = self.build_DQN(nn_args)
            self.networks['learning_scheme'] = 'DQN'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete')
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'DDQN':
            nn_args = {
                'id':self.id, 
                'lr':self.lr, 
                'output_size':self.output_size, 
                'input_size':self.network_input_size,
            }
            self.networks = self.build_DDQN(nn_args)
            self.networks['learning_scheme'] = 'DDQN'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete')
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'DDPG':
            actor_nn_args = {
                'id':self.id,
                'output_size':self.output_size,
                'input_size':self.network_input_size,
                'lr': self.lr,
                'min_max_action':self.min_max_action}
            critic_nn_args = {
                'id':self.id,
                'output_size':self.output_size,
                'input_size':self.network_input_size + actor_nn_args['output_size'],
                'lr': self.lr
                }
            self.networks = self.build_DDPG(actor_nn_args, critic_nn_args)
            self.networks['learning_scheme'] = 'DDPG'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous')
            self.networks['output_size'] = self.output_size
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == "RDDPG":
            lstm_nn_args = {
                'lr': self.lr,
                'input_size': self.network_input_size,
                'output_size': self.meta_param_size,
                'embedding_size': self.recurrent_embedding_size,
                'hidden_size': self.recurrent_hidden_size,
                'num_layers': self.recurrent_num_layers,
                'batch_size': self.batch_size
            }
            actor_nn_args = {
                'id':self.id,
                'output_size':self.output_size,
                'input_size':lstm_nn_args['output_size'],
                'lr': self.lr,
                'min_max_action':self.min_max_action}
            critic_nn_args = {
                'id':self.id,
                'output_size':self.output_size,
                'input_size':lstm_nn_args['output_size'] + actor_nn_args['output_size'],
                'lr': self.lr
                }
            self.networks = self.build_RDDPG()
            self.networks['learning_scheme'] = 'RDDPG'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous')
            self.networks['output_size'] = self.output_size
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'TD3':
            actor_nn_args = {
                'id':self.id,
                'alpha':self.alpha,
                'input_size': self.network_input_size,
                'fc1_dims':400,
                'fc2_dims':300,
                'output_size':self.output_size,
                'min_max_action': self.min_max_action
            }
            critic_nn_args = {
                'id':self.id,
                'beta':self.beta,
                'input_size':self.network_input_size+actor_nn_args['output_size'],
                'fc1_dims':400,
                'fc2_dims':300,
                'output_size':self.output_size}
            self.networks = self.build_TD3(actor_nn_args, critic_nn_args)
            self.networks['learning_scheme'] = 'TD3'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous')
            self.networks['output_size'] = self.output_size
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

    def build_RDDPG(self, lstm_nn_args, actor_nn_args, critic_nn_args):
        return self.make_RDDPG_networks(lstm_nn_args, actor_nn_args, critic_nn_args)

    def build_TD3(self, actor_nn_args, critic_nn_args):
        return self.make_TD3_networks(actor_nn_args, critic_nn_args)

    def build_gsp_network(self, learning_scheme:str | None =None):
        self.gsp_networks = None
        if self.attention_gsp:
            nn_args = {
                'input_size': self.gsp_network_input,
                'output_size': self.gsp_network_output,
                'min_max_action': self.gsp_min_max_action,
                'encode_size': 2,
                'embed_size':256, 
                'hidden_size':256,
                'heads':8,
                'forward_expansion':4,
                'dropout':0,
                'max_length':self.gsp_sequence_length
            }
            self.gsp_networks = self.make_Attention_Encoder(nn_args)
            self.gsp_networks['learning_scheme'] = 'attention'
            self.gsp_networks['replay'] = AttentionSequenceReplayBuffer(num_observations = self.gsp_network_input, seq_len = 5)
            self.gsp_networks['learn_step_counter'] = 0
        else:
            if learning_scheme == 'DDPG':
                if self.recurrent_gsp:
                    self.gsp_networks = self.build_RDDPG_gsp()
                    self.gsp_networks['learning_scheme'] = 'RDDPG'
                    self.gsp_networks['output_size'] = self.gsp_network_output
                    #self.gsp_networks['replay'] = ReplayBuffer(self.mem_size, self.gsp_network_input, 1, 'Continuous', use_gsp = True)
                    self.gsp_networks['replay'] = SequenceReplayBuffer(self.mem_size, self.gsp_network_input, self.gsp_network_output, self.gsp_sequence_length)
                    #SequenceReplayBuffer(max_sequence=100, num_observations = self.gsp_network_input, num_actions = 1, seq_len = 5)
                    self.gsp_networks['learn_step_counter'] = 0
                else:
                    self.gsp_networks = self.build_DDPG_gsp()
                    self.gsp_networks['learning_scheme'] = 'DDPG'
                    self.gsp_networks['output_size'] = self.gsp_network_output
                    self.gsp_networks['replay'] = ReplayBuffer(self.mem_size, self.gsp_network_input, self.gsp_network_output, 'Continuous')
                    self.gsp_networks['learn_step_counter'] = 0
            elif learning_scheme == 'TD3':
                if self.recurrent_gsp:
                    self.gsp_networks = self.build_RTD3_gsp()
                    self.gsp_networks['learning_scheme'] = 'RTD3'
                    self.gsp_networks['output_size']  = self.gsp_network_output
                    self.gsp_networks['replay'] = SequenceReplayBuffer(max_sequence=100, num_observations = self.gsp_network_input, num_actions = self.gsp_network_output, seq_len = 5)
                    self.gsp_networks['learn_step_counter'] = 0
                else:
                    self.gsp_networks = self.build_TD3_gsp()
                    self.gsp_networks['learning_scheme'] = 'TD3'
                    self.gsp_networks['output_size'] = 1
                    self.gsp_networks['replay'] = ReplayBuffer(self.mem_size, self.gsp_network_input, self.gsp_network_output, 'Continuous')
                    self.gsp_networks['learn_step_counter'] = 0
            else:
                raise Exception('[Error] gsp learning scheme is not recognised: '+learning_scheme)

    def build_DDPG_gsp(self):
        actor_nn_args = {
            'id':self.id,
            'input_size':self.gsp_network_input,
            'output_size':self.gsp_network_output,
            'lr': self.lr,
            'min_max_action':self.min_max_action
        }
        critic_nn_args = {
            'id':self.id,
            'input_size':self.gsp_network_input+self.gsp_network_output,
            'output_size': 1,
            'lr': self.lr
        }
        return self.make_DDPG_networks(actor_nn_args, critic_nn_args)
    
    def build_RDDPG_gsp(self):
        lstm_nn_args = {
            'lr': self.lr,
            'input_size': self.gsp_network_input,
            'output_size': self.meta_param_size,
            'embedding_size': self.recurrent_embedding_size,
            'hidden_size': self.recurrent_hidden_size,
            'num_layers': self.recurrent_num_layers,
            'batch_size': self.batch_size
        }
        actor_nn_args = {
            'id':self.id,
            'input_size':lstm_nn_args['output_size'],
            'output_size':self.gsp_network_output,
            'lr': self.lr,
            'min_max_action':self.min_max_action
        }
        critic_nn_args = {
            'id':self.id,
            'input_size':lstm_nn_args['output_size']+actor_nn_args['output_size'],
            'output_size': 1,
            'lr': self.lr
        }
        # print('[INPUT]: ', lstm_nn_args['input_size'])
        # print('[LSTM OUTPUT]', lstm_nn_args['output_size'])
        # print('[DDPG INPUT]', actor_nn_args['input_size'])
        # print('[DDPG OUTPUT]', actor_nn_args['output_size'])
        return self.make_RDDPG_networks(lstm_nn_args, actor_nn_args, critic_nn_args)

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        # Update gsp Networks 
        if self.gsp:
            if self.gsp_networks['learning_scheme'] == 'DDPG' or self.gsp_networks['learning_scheme'] == 'RDDPG':
                self.gsp_networks = self.update_DDPG_network_parameters(tau, self.gsp_networks)
            elif self.gsp_networks['learning_scheme'] == 'TD3' or self.gsp_networks['learning_scheme'] == 'RTD3':
                self.gsp_networks = self.update_TD3_network_parameters(tau, self.gsp_networks)
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
                actions+=T.normal(0.0, self.noise, size = (1, networks['output_size'])).to(networks['actor'].device)
            actions = T.clamp(actions, -self.min_max_action, self.min_max_action)
            return actions[0].cpu().detach().numpy()
        elif networks['learning_scheme'] == 'TD3':
            actions = self.TD3_choose_action(observation, networks, self.output_size)
            return actions[0]
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
    
    def learn(self):
        # TODO Not sure why we have n_agents*batch_size + batch_size
        if self.networks['replay'].mem_ctr < self.batch_size: # (self.n_agents*self.batch_size + self.batch_size): 
                return

        if self.gsp:
            if self.networks['learn_step_counter'] % self.gsp_learning_offset == 0:
                #print('[DEBUG] Learning Attention', self.networks['learn_step_counter'])
                self.learn_gsp()

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

    def learn_gsp(self):
        if self.gsp_networks['replay'].mem_ctr < self.gsp_batch_size:
                return
        if self.gsp_networks['learning_scheme'] in {'DDPG'}:
            self.learn_DDPG(self.gsp_networks, self.gsp, self.recurrent_gsp)
        elif self.gsp_networks['learning_scheme'] in {'RDDPG'}:
            self.learn_RDDPG(self.gsp_networks, self.gsp, self.recurrent_gsp)
        elif self.gsp_networks['learning_scheme'] == 'TD3':
            self.learn_TD3(self.gsp_networks, self.gsp, self.recurrent_gsp)
        elif self.gsp_networks['learning_scheme'] == 'attention':
            self.learn_attention(self.gsp_networks)

    def store_agent_transition(self, s, a, r, s_, d):
        self.store_transition(s, a, r, s_, d, self.networks)
    
    def store_gsp_transition(self, s, a, r, s_, d):
        if self.attention_gsp:
            self.store_attention_transition(s, a, self.gsp_networks)
        else:
            self.store_transition(s, a, r, s_, d, self.gsp_networks)

    def reset_gsp_sequence(self):
        self.gsp_sequence = [np.zeros(self.gsp_network_input) for i in range(self.gsp_sequence_length)]
    
    def add_gsp_sequence(self, obs):
        self.gsp_sequence.append(obs)
        self.gsp_sequence.pop(0)

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
        if self.attention_gsp:
            if self.gsp_networks['learning_scheme'] == 'attention':
                self.gsp_networks['attention'].save_checkpoint(path)
        elif self.gsp:
            self.gsp_networks['actor'].save_checkpoint(path, self.gsp)
            self.gsp_networks['target_actor'].save_checkpoint(path, self.gsp)
            if self.gsp_networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
                self.gsp_networks['critic'].save_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic'].save_checkpoint(path, self.gsp)
            elif self.gsp_networks['learning_scheme'] in {'TD3', 'RTD3'}:
                self.gsp_networks['critic_1'].save_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic_1'].save_checkpoint(path, self.gsp)
                self.gsp_networks['critic_2'].save_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic_2'].save_checkpoint(path, self.gsp)

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
        
        if self.attention_gsp:
            if self.gsp_networks['learning_scheme'] == 'attention':
                self.gsp_networks['attention'].load_checkpoint(path)
        elif self.gsp:
            self.gsp_networks['actor'].load_checkpoint(path, self.gsp)
            self.gsp_networks['target_actor'].load_checkpoint(path, self.gsp)
            if self.gsp_networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
                self.gsp_networks['critic'].load_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic'].load_checkpoint(path, self.gsp)
            elif self.gsp_networks['learning_scheme'] in {'TD3', 'RTD3'}:
                self.gsp_networks['critic_1'].load_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic_1'].load_checkpoint(path, self.gsp)
                self.gsp_networks['critic_2'].load_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic_2'].load_checkpoint(path, self.gsp)
        
    
if __name__=='__main__':
    agent_args = {'id':1, 'input_size':32, 'output_size':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'gsp':False, 'recurrent_gsp':False, 'gsp_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['input_size'])

    
    print('[TESTING] DQN')
    agent.build_networks('DQN')
    agent.networks['learn_step_counter'] = agent.replace_target_ctr
    agent.replace_target_network()
    observation = np.random.random(size = agent_args['input_size'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['input_size'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())

    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])

    print('[TESTING] DDQN')
    agent_args = {'id':1, 'input_size':32, 'output_size':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'gsp':False, 'recurrent_gsp':False, 'gsp_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['input_size'])
    agent.build_networks('DDQN')
    agent.networks['learn_step_counter'] = agent.replace_target_ctr
    agent.replace_target_network()
    observation = np.random.random(size = agent_args['input_size'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['input_size'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())
    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])
    
    print('[TESTING] DDPG and param update')
    agent_args = {'id':1, 'input_size':32, 'output_size':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'gsp':False, 'recurrent_gsp':False, 'gsp_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['input_size'])
    agent.build_networks('DDPG')
    agent.update_network_parameters()
    observation = np.random.random(size = agent_args['input_size'])
    done = False
    for i in range(200):
        action = [None, agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['input_size'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())

    print('[TESTING] TD3')
    agent_args = {'id':1, 'input_size':32, 'output_size':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'gsp':False, 'recurrent_gsp':False, 'gsp_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['input_size'])
    agent.build_networks('TD3')
    agent.update_network_parameters()
    observation = np.random.random(size = agent_args['input_size'])
    done = False
    for i in range(200):
        action = [None, agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['input_size'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())
    print(agent.networks['actor'])
    print(agent.networks['critic_1'])
    print(agent.networks['critic_2'])
    

    print('[TESTING] gsp DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, gsp=True, recurrent_gsp = False)
    agent.build_networks('DDPG')
    agent.build_gsp_network('DDPG')
    agent.update_network_parameters()
    print(agent.gsp_networks['actor'])
    print(agent.gsp_networks['critic'])
    

    print('[TESTING] Recurrent gsp DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, gsp=True, recurrent_gsp = True)
    agent.build_networks('DDPG')
    agent.build_gsp_network('DDPG')
    agent.update_network_parameters()
    print(agent.gsp_networks['actor'])
    print(agent.gsp_networks['critic'])
    print(agent.gsp_networks['ee'])