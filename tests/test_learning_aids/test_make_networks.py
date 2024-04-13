import numpy as np
import os
import yaml
from gsp_rl.src.networks import (
    DQN, 
    DDQN,
    DDPGActorNetwork,
    DDPGCriticNetwork,
    RDDPGActorNetwork,
    RDDPGCriticNetwork,
    TD3ActorNetwork,
    TD3CriticNetwork,
    EnvironmentEncoder,
    AttentionEncoder
)

from gsp_rl.src.actors import NetworkAids

containing_folder = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(containing_folder, 'config.yml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

NA = NetworkAids(config)

def test_make_DQN_networks():
    """
    Test the base code tha makes the DQN networks
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
        networks = NA.make_DQN_networks(nn_args)
        for name, param in networks['q_eval'].named_parameters():
            shape = tuple(param.size())
            if name == 'fc1.weights':
                assert(shape == (nn_args['fc1_dims'], nn_args['input_size']))
            if name == 'fc2.weights':
                assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
            if name == 'fc3.weights':
                assert(shape == (nn_args['output_size'], nn_args['fc2_dims']))
        
        for name, param in networks['q_next'].named_parameters():
            shape = tuple(param.size())
            if name == 'fc1.weights':
                assert(shape == (nn_args['fc1_dims'], nn_args['input_size']))
            if name == 'fc2.weights':
                assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
            if name == 'fc3.weights':
                assert(shape == (nn_args['output_size'], nn_args['fc2_dims']))

def test_make_DDQN_networks():
    """
    Test the base code tha makes the DDQN networks
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
        networks = NA.make_DDQN_networks(nn_args)
        for name, param in networks['q_eval'].named_parameters():
            shape = tuple(param.size())
            if name == 'fc1.weights':
                assert(shape == (nn_args['fc1_dims'], nn_args['input_size']))
            if name == 'fc2.weights':
                assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
            if name == 'fc3.weights':
                assert(shape == (nn_args['output_size'], nn_args['fc2_dims']))
        
        for name, param in networks['q_next'].named_parameters():
            shape = tuple(param.size())
            if name == 'fc1.weights':
                assert(shape == (nn_args['fc1_dims'], nn_args['input_size']))
            if name == 'fc2.weights':
                assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
            if name == 'fc3.weights':
                assert(shape == (nn_args['output_size'], nn_args['fc2_dims']))

def test_make_DDPG_networks():
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
            'input_size':actor_nn_args['input_size'] + actor_nn_args['output_size'],
            'output_size':1,
            'fc1_dims': 400,
            'fc2_dims':300
        }
        networks = NA.make_DDPG_networks(actor_nn_args, critic_nn_args)
        for name, param in networks['actor'].named_parameters():
            shape = param.shape
            if name == 'fc1.weight':
                assert(shape == (actor_nn_args['fc1_dims'], actor_nn_args['input_size']))
            elif name == 'fc2.weight':
                assert(shape == (actor_nn_args['fc2_dims'], actor_nn_args['fc1_dims']))
            elif name == 'fc3.weight':
                assert(shape == (actor_nn_args['output_size'], actor_nn_args['fc2_dims']))

        for name, param in networks['critic'].named_parameters():
            shape = param.shape
            if name == 'fc1.weight':
                assert(shape == (critic_nn_args['fc1_dims'], critic_nn_args['input_size']))
            elif name == 'fc2.weight':
                assert(shape == (critic_nn_args['fc2_dims'], critic_nn_args['fc1_dims']))
            elif name == 'fc3.weight':
                assert(shape == (1, critic_nn_args['fc2_dims']))

def test_make_RDDPG_networks():
    """
    Test the base code tha makes the DDPG networks
    """
    for i in range(10):
        lstm_nn_args = {
            'lr':1e-5,
            'input_size':np.random.randint(4, 50),
            'output_size':256,
            'embedding_size':200,
            'hidden_size':300,
            'num_layers':5,
            'batch_size':16
        }

        ddpg_actor_nn_args = {
            'id': 1,
            'lr': 1e-4,
            'input_size':lstm_nn_args['output_size'],
            'output_size': np.random.randint(2, 30),
            'fc1_dims': 400,
            'fc2_dims': 300
        }
        ddpg_critic_nn_args = {
            'id': 1,
            'lr': 1e-4,
            'input_size':lstm_nn_args['output_size'] + ddpg_actor_nn_args['output_size'],
            'output_size': 1,
            'fc1_dims': 400,
            'fc2_dims': 300
        }

        networks = NA.make_RDDPG_networks(lstm_nn_args, ddpg_actor_nn_args, ddpg_critic_nn_args)
        for name, param in networks['actor'].named_parameters():
            shape = param.shape
            if name == 'ee.embedding.wight':
                assert(shape[0] == lstm_nn_args['embedding_size'])
                assert(shape[1] == lstm_nn_args['input_size'])
            elif name == 'ee.meta_layer.weight':
                assert(shape[0] == lstm_nn_args['output_size'])
                assert(shape[1] == lstm_nn_args['hidden_size'])
            elif name == 'actor.fc1.weight':
                assert(shape[1] == lstm_nn_args['output_size'])
            elif name == 'actor.mu.weight':
                assert(shape[0] == ddpg_actor_nn_args['output_size'])

        for name, param in networks['critic'].named_parameters():
            shape = param.shape
            if name == 'ee.embedding.wight':
                assert(shape[0] == lstm_nn_args['embedding_size'])
                assert(shape[1] == lstm_nn_args['input_size'])
            elif name == 'ee.meta_layer.weight':
                assert(shape[0] == lstm_nn_args['output_size'])
                assert(shape[1] == lstm_nn_args['hidden_size'])
            elif name == 'critic.fc1.weight':
                assert(shape[1] == ddpg_critic_nn_args['input_size'])
            elif name == 'critic.q.weight':
                assert(shape[0] == 1)

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
            'input_size':actor_nn_args['input_size']+actor_nn_args['output_size'],
            'output_size':actor_nn_args['output_size'],
            'fc1_dims': 400,
            'fc2_dims':300
        }

        networks = NA.make_TD3_networks(actor_nn_args, critic_nn_args)
        for name, param in networks['actor'].named_parameters():
            shape = param.shape
            if name == 'fc1.weight':
                assert(shape == (actor_nn_args['fc1_dims'], actor_nn_args['input_size']))
            elif name == 'fc2.weight':
                assert(shape == (actor_nn_args['fc2_dims'], actor_nn_args['fc1_dims']))
            elif name == 'fc3.weight':
                assert(shape == (actor_nn_args['output_size'], actor_nn_args['fc2_dims']))

        for name, param in networks['critic_1'].named_parameters():
            shape = param.shape
            if name == 'fc1.weight':
                assert(shape == (critic_nn_args['fc1_dims'], actor_nn_args['input_size']+actor_nn_args['output_size']))
            elif name == 'fc2.weight':
                assert(shape == (critic_nn_args['fc2_dims'], critic_nn_args['fc1_dims']))
            elif name == 'fc3.weight':
                assert(shape == (1, critic_nn_args['fc2_dims']))

        for name, param in networks['critic_2'].named_parameters():
            shape = param.shape
            if name == 'fc1.weight':
                assert(shape == (critic_nn_args['fc1_dims'], actor_nn_args['input_size']+actor_nn_args['output_size']))
            elif name == 'fc2.weight':
                assert(shape == (critic_nn_args['fc2_dims'], critic_nn_args['fc1_dims']))
            elif name == 'fc3.weight':
                assert(shape == (1, critic_nn_args['fc2_dims']))

def test_make_Environmental_Encoder():
    lstm_nn_args = {
            'lr':1e-5,
            'input_size':np.random.randint(4, 50),
            'output_size':256,
            'embedding_size':200,
            'hidden_size':300,
            'num_layers':5,
            'batch_size':16
        }
    networks = NA.make_Environmental_Encoder(lstm_nn_args)
    for name, param in networks['ee'].named_parameters():
        shape = param.shape
        if name == 'ee.embedding.wight':
            assert(shape[0] == lstm_nn_args['embedding_size'])
            assert(shape[1] == lstm_nn_args['input_size'])
        elif name == 'ee.meta_layer.weight':
            assert(shape[0] == lstm_nn_args['output_size'])
            assert(shape[1] == lstm_nn_args['hidden_size'])

def test_make_Attention_Encoder():
    """
    Test the base code tha makes the Attention Encoder
    """
    for i in range(10):
        nn_args = {
            'input_size':np.random.randint(4, 15),
            'output_size':np.random.randint(1, 7),
            'min_max_action': 1,
            'encode_size':1,
            'embed_size':256,
            'hidden_size':256,
            'heads': 8,
            'forward_expansion': 4,
            'dropout': 0,
            'max_length': 5
        }
        networks = NA.make_Attention_Encoder(nn_args)
        for name, param in networks['attention'].named_parameters():
            shape = tuple(param.size())
            if name == 'word_embedding.0.weight':
                assert(shape == (nn_args['hidden_size'], nn_args['input_size']))
            if name == 'word_embedding.2.weight':
                assert(shape == (1, nn_args['hidden_size']))
            if name == 'position_embedding.weight':
                assert(shape == (nn_args['max_length'], nn_args['embed_size']))
            if name == 'fc_out.weight':
                assert(shape == (nn_args['output_size'], nn_args['max_length']*nn_args['embed_size']))
    