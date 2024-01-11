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

def test_make_DQN_networks():
    """
    Test the base code tha makes the DQN networks
    """
    nn_args = {
        'id':1,
        'lr':1e-4,
        'input_size':40,
        'output_size':9,
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
    nn_args = {
        'id':1,
        'lr':1e-4,
        'input_size':40,
        'output_size':9,
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
    actor_nn_args = {
        'id':1,
        'lr':1e-4,
        'input_size':40,
        'output_size':2,
        'fc1_dims': 400,
        'fc2_dims':300
    }

    critic_nn_args = {
        'id':1,
        'lr':1e-4,
        'input_size':40,
        'actor_output_size':actor_nn_args['output_size'],
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
            assert(shape == (actor_nn_args['fc1_dims'], actor_nn_args['input_size']+critic_nn_args['actor_output_size']))
        elif name == 'fc2.weight':
            assert(shape == (actor_nn_args['fc2_dims'], actor_nn_args['fc1_dims']))
        elif name == 'fc3.weight':
            assert(shape == (1, actor_nn_args['fc2_dims']))