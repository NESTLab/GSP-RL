from gsp_rl.src.networks import (
    EnvironmentEncoder,
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


def test_building_recurrent_actor_network():
    ee = EnvironmentEncoder(**lstm_nn_args)
    for name, param in ee.named_parameters():
        shape = param.shape
        if name == 'embedding.wight':
            assert(shape[0] == lstm_nn_args['embedding_size'])
            assert(shape[1] == lstm_nn_args['input_size'])
        elif name == 'meta_layer.weight':
            assert(shape[0] == lstm_nn_args['output_size'])
            assert(shape[1] == lstm_nn_args['hidden_size'])


def test_actor_forward():
    ee = EnvironmentEncoder(**lstm_nn_args)
    testing_data = [T.randn((lstm_nn_args['input_size'])) for _ in range(10)]
    testing_data = T.tensor(np.array(testing_data)).to(ee.device)
    assert(ee(testing_data).shape[-1] == lstm_nn_args['output_size'])