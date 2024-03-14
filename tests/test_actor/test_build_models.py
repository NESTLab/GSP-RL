from gsp_rl.src.actors import Actor

def test_build_networks_DQN():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'intention':False,
            'recurrent_intention':False,
            'attention': False,
            'intention_input_size': 6,
            'intention_output_size': 1,
            'intention_look_back':2,
            'intention_sequence_length': 5
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
            'intention':False,
            'recurrent_intention':False,
            'attention': False,
            'intention_input_size': 6,
            'intention_output_size': 1,
            'intention_look_back':2,
            'intention_sequence_length': 5
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
            'intention':False,
            'recurrent_intention':False,
            'attention': False,
            'intention_input_size': 6,
            'intention_output_size': 1,
            'intention_look_back':2,
            'intention_sequence_length': 5
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
            'intention':False,
            'recurrent_intention':False,
            'attention': False,
            'intention_input_size': 6,
            'intention_output_size': 1,
            'intention_look_back':2,
            'intention_sequence_length': 5
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

def test_build_intention_networks_DDPG():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'intention':True,
            'recurrent_intention':False,
            'attention': False,
            'intention_input_size': 6,
            'intention_output_size': 1,
            'intention_look_back':2,
            'intention_sequence_length': 5
    }
    actor = Actor(**nn_args)
    networks = actor.intention_networks
    for name, param in networks['actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'mu.weights':
            assert(shape == nn_args['output_size'])
        
    for name, param in networks['critic'].named_parameters():
        shape = tuple(param.size())
        if name == 'q.weights':
            assert(shape == 1)

def test_build_intention_networks_Attention():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2, 
            'intention':True,
            'recurrent_intention':False,
            'attention': True,
            'intention_input_size': 6,
            'intention_output_size': 1,
            'intention_look_back':2,
            'intention_sequence_length': 5
    }
    actor = Actor(**nn_args)
    networks = actor.intention_networks
    for name, param in networks['attention'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc_out.weights':
            assert(shape == nn_args['output_size'])

  