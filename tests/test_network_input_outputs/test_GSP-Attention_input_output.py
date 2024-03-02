from gsp_rl.src.networks import (
    AttentionEncoder
)

import torch as T

def test_building_attention_network():
    nn_args = {
        'input_size': 10,
        'output_size': 1,
        'min_max_action': 1.0,
        'encode_size': 1,
        'embed_size': 256,
        'hidden_size': 256,
        'heads': 8,
        'forward_expansion': 4,
        'dropout': 0,
        'max_length': 5,
    }
    attention_network = AttentionEncoder(**nn_args)
    for name, param in attention_network.named_parameters():
        shape = tuple(param.size())
        if name == 'word_embedding.0.weight':
            assert(shape == (nn_args['hidden_size'], nn_args['input_size']))
        if name == 'word_embedding.2.weight':
            assert(shape == (1, nn_args['hidden_size']))
        if name == 'position_embedding.weight':
            assert(shape == (nn_args['max_length'], nn_args['embed_size']))
        if name == 'fc_out.weight':
            assert(shape == (nn_args['output_size'], nn_args['max_length']*nn_args['embed_size']))

def test_attention_forward():
    nn_args = {
        'input_size': 10,
        'output_size': 1,
        'min_max_action': 1.0,
        'encode_size': 1,
        'embed_size': 256,
        'hidden_size': 256,
        'heads': 8,
        'forward_expansion': 4,
        'dropout': 0,
        'max_length': 5,
    }
    attention_network = AttentionEncoder(**nn_args)
    random_input = T.rand(1, nn_args['max_length'], nn_args['input_size'])
    assert(attention_network(random_input).shape[1] == nn_args['output_size'])
        