from gsp_rl.src.networks import (
    TD3ActorNetwork,
    TD3CriticNetwork,
)

import torch as T

def test_building_actor_network():
    id: int = 1
    alpha: float = 1e-4
    input_size: int = 30
    output_size: int = 3
    fc1_dims:int  = 200
    fc2_dims: int = 400
    DQN_Network = TD3ActorNetwork(id, alpha, input_size, output_size, fc1_dims, fc2_dims)
    for name, param in DQN_Network.named_parameters():
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
    alpha: float = 1e-4
    input_size: int = 30
    output_size: int = 3
    fc1_dims:int  = 200
    fc2_dims: int = 400
    TD3_Actor_Network = TD3ActorNetwork(id, alpha, input_size, output_size, fc1_dims, fc2_dims)
    random_observation = T.rand((1, input_size))
    assert(TD3_Actor_Network(random_observation).shape[1] == output_size)

def test_building_critic_network():
    id: int = 1
    beta: float = 1e-4
    input_size: int = 30
    actor_output_size: int = 2
    fc1_dims:int  = 200
    fc2_dims: int = 400
    DQN_Network = TD3CriticNetwork(id, beta, input_size, actor_output_size, fc1_dims, fc2_dims)
    for name, param in DQN_Network.named_parameters():
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
    beta: float = 1e-4
    input_size: int = 30
    actor_output_size: int = 2
    fc1_dims:int  = 200
    fc2_dims: int = 400
    TD3_Critic_Network = TD3CriticNetwork(id, beta, input_size, actor_output_size, fc1_dims, fc2_dims)
    random_input = T.rand((1, input_size))
    random_action = T.rand((1, actor_output_size))
    assert(TD3_Critic_Network(random_input, random_action).shape[1] == 1)
    