import numpy as np
import torch as T
from gsp_rl.src.networks import (
    DQN, 
    DDQN,
    DDPGActorNetwork,
    DDPGCriticNetwork,
    TD3ActorNetwork,
    TD3CriticNetwork,
    AttentionEncoder
)

from gsp_rl.src.actors import NetworkAids

NA = NetworkAids()

def test_DQN_DDQN_choose_action():
    """
    Test the choose action function for DQN and DDQN
    """
    for i in range(10):
        nn_args = {
            'id':1,
            'lr':1e-4,
            'input_size':np.random.randint(4, 50),
            'output_size':np.random.randint(1, 15),
            'fc1_dims': 64,
            'fc2_dims':128
        }
        DQN_networks = NA.make_DQN_networks(nn_args)
        DDQN_networks = NA.make_DDQN_networks(nn_args)

        random_input = np.random.uniform(0, 1, nn_args['input_size'])
        assert(NA.DQN_DDQN_choose_action(random_input, DQN_networks) in range(nn_args['output_size']))
        assert(NA.DQN_DDQN_choose_action(random_input, DDQN_networks) in range(nn_args['output_size']))
        

def test_DDPG_choose_action():
    """
    Test the base code tha makes the DDPG networks
    """
    for i in range(10):
        actor_nn_args = {
            'id':1,
            'lr':1e-4,
            'input_size':np.random.randint(4, 50),
            'output_size':np.random.randint(1, 15),
            'fc1_dims': 400,
            'fc2_dims':300
        }

        critic_nn_args = {
            'id':1,
            'lr':1e-4,
            'input_size':actor_nn_args['input_size'],
            'actor_output_size':actor_nn_args['output_size'],
            'fc1_dims': 400,
            'fc2_dims':300
        }

        networks = NA.make_DDPG_networks(actor_nn_args, critic_nn_args)
        networks['learning_scheme'] = 'DDPG'
        random_input = np.random.uniform(0, 1, actor_nn_args['input_size'])
        assert(tuple(NA.DDPG_choose_action(random_input, networks).shape) == (1, actor_nn_args['output_size']))

def test_make_TD3_networks():
    """
    Test the base code tha makes the TD3 networks
    """
    for i in range(10):
        actor_nn_args = {
            'id':1,
            'alpha':1e-4,
            'input_size':np.random.randint(4, 40),
            'output_size':np.random.randint(1, 15),
            'fc1_dims': 400,
            'fc2_dims':300
        }

        critic_nn_args = {
            'id':1,
            'beta':1e-4,
            'input_size':actor_nn_args['input_size'],
            'actor_output_size':actor_nn_args['output_size'],
            'fc1_dims': 400,
            'fc2_dims':300
        }

        networks = NA.make_TD3_networks(actor_nn_args, critic_nn_args)
        random_input = np.random.uniform(0, 1, actor_nn_args['input_size'])
        assert(tuple(NA.TD3_choose_action(random_input, networks, actor_nn_args['output_size']).shape) == (1, actor_nn_args['output_size']))

def test_Attention_choose_action():
    """
    Test the base code tha makes the Attention Encoder
    """
    for i in range(10):
        nn_args = {
            'input_size':np.random.randint(4, 15),
            'output_size':np.random.randint(1, 7),
            'encode_size':1,
            'embed_size':256,
            'hidden_size':256,
            'heads': 8,
            'forward_expansion': 4,
            'dropout': 0,
            'max_length': 5
        }
        networks = NA.make_Attention_Encoder(nn_args)
        random_input = np.random.uniform(0, 1, (1, nn_args['max_length'],nn_args['input_size']))
        
        assert(tuple(NA.Attention_choose_action(random_input, networks).shape) == (1, nn_args['output_size']))