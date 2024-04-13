from gsp_rl.src.actors import Actor
import os
import yaml

containing_folder = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(containing_folder, 'config.yml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def test_build_networks_DQN():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.networks
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

def test_build_networks_DDQN():
    nn_args = {
            'id':1,
            'network': 'DDQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.networks
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

def test_build_networks_DDPG():
    nn_args = {
            'id':1,
            'network': 'DDPG',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.networks
    for name, param in networks['actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (nn_args['output_size'], nn_args['fc2_dims']))
    
    for name, param in networks['target_actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (nn_args['output_size'], nn_args['fc2_dims']))
    
    for name, param in networks['critic'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (1, nn_args['fc2_dims']))
    
    for name, param in networks['target_critic'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (1, nn_args['fc2_dims']))


def test_build_networks_TD3():
    nn_args = {
            'id':1,
            'network': 'TD3',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.networks
    for name, param in networks['actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (nn_args['output_size'], nn_args['fc2_dims']))
    
    for name, param in networks['target_actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (nn_args['output_size'], nn_args['fc2_dims']))
    
    for name, param in networks['critic_1'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (1, nn_args['fc2_dims']))
    
    for name, param in networks['target_critic_1'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (1, nn_args['fc2_dims']))

    for name, param in networks['critic_2'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (1, nn_args['fc2_dims']))
    
    for name, param in networks['target_critic_2'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weights':
            assert(shape == (nn_args['fc1_dims'], nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weights':
            assert(shape == (nn_args['fc2_dims'], nn_args['fc1_dims']))
        if name == 'fc3.weights':
            assert(shape == (1, nn_args['fc2_dims']))

def test_build_gsp_networks_DDPG():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'gsp':True,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.gsp_networks
    for name, param in networks['actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'mu.weights':
            assert(shape == nn_args['output_size'])
        
    for name, param in networks['critic'].named_parameters():
        shape = tuple(param.size())
        if name == 'q.weights':
            assert(shape == 1)

def test_build_gsp_networks_Attention():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'gsp':True,
            'recurrent_gsp':False,
            'attention': True,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.gsp_networks
    for name, param in networks['attention'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc_out.weights':
            assert(shape == nn_args['output_size'])

  