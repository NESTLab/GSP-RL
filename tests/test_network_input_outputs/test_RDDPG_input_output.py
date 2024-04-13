from gsp_rl.src.networks import (
    DDPGActorNetwork,
    DDPGCriticNetwork,
    EnvironmentEncoder,
    RDDPGActorNetwork,
    RDDPGCriticNetwork,
)

import torch as T
import numpy as np

lstm_nn_args = {
    'lr':1e-5,
    'input_size':5,
    'output_size':100,
    'embedding_size':256,
    'hidden_size':256,
    'num_layers':5,
    'batch_size':16
}

ddpg_actor_nn_args = {
    'id': 1,
    'lr': 1e-4,
    'input_size':lstm_nn_args['output_size'],
    'output_size': 2,
    'fc1_dims': 400,
    'fc2_dims': 300
}
ddpg_critic_nn_args = {
    'id': 1,
    'lr': 1e-4,
    'input_size':lstm_nn_args['output_size'] + ddpg_actor_nn_args['output_size'],
    'output_size': 2,
    'fc1_dims': 400,
    'fc2_dims': 300
}

def test_building_recurrent_actor_network():
    ee = EnvironmentEncoder(**lstm_nn_args)
    actor = DDPGActorNetwork(**ddpg_actor_nn_args)
    rddpg_actor = RDDPGActorNetwork(ee, actor)
    for name, param in rddpg_actor.named_parameters():
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


def test_actor_forward():
    ee = EnvironmentEncoder(**lstm_nn_args)
    actor = DDPGActorNetwork(**ddpg_actor_nn_args)
    rddpg_actor = RDDPGActorNetwork(ee, actor)
    testing_data = [T.randn((lstm_nn_args['input_size'])) for _ in range(10)]
    testing_data = T.tensor(np.array(testing_data)).to(rddpg_actor.device)
    assert(rddpg_actor(testing_data).shape[-1] == ddpg_actor_nn_args['output_size'])

def test_building_critic_network():
    ee = EnvironmentEncoder(**lstm_nn_args)
    critic = DDPGCriticNetwork(**ddpg_critic_nn_args)
    rddpg_critic = RDDPGCriticNetwork(ee, critic)
    for name, param in rddpg_critic.named_parameters():
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

def test_critic_forward():
    ee = EnvironmentEncoder(**lstm_nn_args)
    actor = DDPGActorNetwork(**ddpg_actor_nn_args)
    critic = DDPGCriticNetwork(**ddpg_critic_nn_args)
    rddpg_actor = RDDPGActorNetwork(ee, actor)
    rddpg_critic = RDDPGCriticNetwork(ee, critic)
    testing_data = [T.randn((lstm_nn_args['input_size'])) for _ in range(10)]
    testing_data = T.tensor(np.array(testing_data)).to(rddpg_critic.device)
    action = rddpg_actor(testing_data)
    value = rddpg_critic(testing_data, action)
    assert(value.shape[-1] == 1)
    