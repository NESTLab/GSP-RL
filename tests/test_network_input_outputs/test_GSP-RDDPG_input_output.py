from gsp_rl.src.networks import (
    RDDPGActorNetwork,
    RDDPGCriticNetwork,
    DDPGActorNetwork,
    DDPGCriticNetwork,
    EnvironmentEncoder,
)

import torch as T
import numpy as np

lstm_nn_args = {
    'lr':1e-5,
    'input_size':25,
    'output_size':100,
    'embedding_size':256,
    'hidden_size':256,
    'num_layers':1,
    'batch_size':16
}

ddpg_actor_nn_args = {
    'id': 1,
    'lr': 1e-4,
    'input_size':lstm_nn_args['output_size'],
    'output_size': 1,
    'fc1_dims': 200,
    'fc2_dims': 400
}

ddpg_critic_nn_args = {
    'id': 1,
    'lr': 1e-4,
    'input_size':lstm_nn_args['output_size'] + ddpg_actor_nn_args['output_size'],
    'output_size': 1,
    'fc1_dims': 200,
    'fc2_dims': 400
}


def test_building_actor_network():
    ee = EnvironmentEncoder(**lstm_nn_args)
    actor = DDPGActorNetwork(**ddpg_actor_nn_args)
    rddpg_actor = RDDPGActorNetwork(ee, actor)
    for name, param in rddpg_actor.named_parameters():
        shape = param.shape
        if name == 'ee.embedding.weight':
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
    random_observation = T.rand((lstm_nn_args['batch_size'], lstm_nn_args['input_size'])).to(rddpg_actor.device)
    output, _ = rddpg_actor(random_observation)
    assert(output.shape[-1] == ddpg_actor_nn_args['output_size'])

def test_building_critic_network():
    ee = EnvironmentEncoder(**lstm_nn_args)
    critic = DDPGCriticNetwork(**ddpg_critic_nn_args)
    rddpg_critic = RDDPGCriticNetwork(ee, critic)
    for name, param in rddpg_critic.named_parameters():
        shape = param.shape
        if name == 'ee.embedding.weight':
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
    random_observation = T.rand((lstm_nn_args['batch_size'], lstm_nn_args['input_size'])).to(rddpg_critic.device)
    action, _ = rddpg_actor(random_observation)
    value, _ = rddpg_critic(random_observation, action)
    assert(value.shape[-1] == 1)
