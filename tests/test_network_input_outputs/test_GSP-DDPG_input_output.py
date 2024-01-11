from gsp_rl.src.networks import (
    DDPGActorNetwork,
    DDPGCriticNetwork,
)

import torch as T

def test_building_actor_network():
    id: int = 1
    lr: float = 1e-4
    input_size: int = 30
    output_size: int = 1
    fc1_dims:int  = 200
    fc2_dims: int = 400
    DDPG_Actor_Network = DDPGActorNetwork(id, lr, input_size, output_size, fc1_dims, fc2_dims)
    for name, param in DDPG_Actor_Network.named_parameters():
        shape = param.shape
        if name == 'fc1.weight':
            assert(shape[1] == input_size)
            assert(shape[0] == fc1_dims)
        elif name == 'fc2.weight':
            assert(shape[1] == fc1_dims)
            assert(shape[0] == fc2_dims)
        elif name == 'fc3.weight':
            assert(shape[1] == fc2_dims)
            assert(shape[0] == output_size)

def test_actor_forward():
    id: int = 1
    lr: float = 1e-4
    input_size: int = 30
    output_size: int = 1
    fc1_dims:int  = 200
    fc2_dims: int = 400
    DDPG_Actor_Network = DDPGActorNetwork(id, lr, input_size, output_size, fc1_dims, fc2_dims)
    random_observation = T.rand((1, input_size))
    assert(DDPG_Actor_Network(random_observation).shape[1] == output_size)

def test_building_critic_network():
    id: int = 1
    lr: float = 1e-4
    input_size: int = 30
    actor_output_size: int = 1
    fc1_dims:int  = 200
    fc2_dims: int = 400
    DDPG_Critic_Network = DDPGCriticNetwork(id, lr, input_size, actor_output_size, fc1_dims, fc2_dims)
    for name, param in DDPG_Critic_Network.named_parameters():
        shape = param.shape
        if name == 'fc1.weight':
            assert(shape[1] == input_size+actor_output_size)
            assert(shape[0] == fc1_dims)
        elif name == 'fc2.weight':
            assert(shape[1] == fc1_dims)
            assert(shape[0] == fc2_dims)
        elif name == 'fc3.weight':
            assert(shape[1] == fc2_dims)
            assert(shape[0] == 1)

def test_critic_forward():
    id: int = 1
    lr: float = 1e-4
    input_size: int = 30
    actor_output_size: int = 1
    fc1_dims:int  = 200
    fc2_dims: int = 400
    DDPG_Critic_Network = DDPGCriticNetwork(id, lr, input_size, actor_output_size, fc1_dims, fc2_dims)
    random_input = T.rand((1, input_size))
    random_action = T.rand((1, actor_output_size))
    assert(DDPG_Critic_Network(random_input, random_action).shape[1] == 1)
    