from gsp_rl.src.networks import (
    DDPGActorNetwork,
    DDPGCriticNetwork,
)

import torch as T

def test_building_network():
    id: int = 1
    lr: float = 1e-4
    input_size: int = 2
    output_size: int = 30
    fc1_dims:int  = 200
    fc2_dims: int = 400
    DQN_Network = DQN(id, lr, input_size, output_size, fc1_dims, fc2_dims)
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

def test_forward():
    id: int = 1
    lr: float = 1e-4
    input_size: int = 3
    output_size: int = 30
    fc1_dims:int  = 200
    fc2_dims: int = 400
    DQN_Network = DQN(id, lr, input_size, output_size, fc1_dims, fc2_dims)
    random_observation = T.rand((1, input_size))
    assert(DQN_Network(random_observation).shape[1] == output_size)
    