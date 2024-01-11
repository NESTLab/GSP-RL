from gsp_rl.src.networks import (
    AttentionEncoder
)

import torch as T

def test_building_attention_network():
    input_size = 10
    output_size = 1
    encode_size = 2
    embed_size = 256
    hidden_size = 256
    heads = 8
    forward_expansion = 4
    dropout = 0
    max_length = 5
    attention_network = AttentionEncoder(
        input_size,
        output_size,
        encode_size,
        embed_size, 
        hidden_size,
        heads, 
        forward_expansion, 
        dropout, 
        max_length
    )
    for name, param in attention_network.named_parameters():
        shape = tuple(param.size())
        print(name, shape)
        if name == 'word_embedding.0.weight':
            assert(shape == (hidden_size, input_size))
        if name == 'word_embedding.2.weight':
            assert(shape == (encode_size, hidden_size))
        if name == 'position_embedding.weight':
            assert(shape == (max_length, embed_size))
        if name == 'fc_out.weight':
            assert(shape == (output_size, max_length*embed_size))

def test_attention_forward():
    input_size = 10
    output_size = 1
    encode_size = 1
    embed_size = 256
    hidden_size = 256
    heads = 8
    forward_expansion = 4
    dropout = 0
    max_length = 5
    attention_network = AttentionEncoder(
        input_size,
        output_size,
        encode_size,
        embed_size, 
        hidden_size,
        heads, 
        forward_expansion, 
        dropout, 
        max_length
    )
    random_input = T.rand(1, max_length, input_size)
    assert(attention_network(random_input).shape[1] == output_size)
        