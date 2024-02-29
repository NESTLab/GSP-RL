from gsp_rl.src.actors import Actor

def test_build_networks_DQN():
    nn_args = {
            'id':1,
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
    actor.build_networks('DQN')
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