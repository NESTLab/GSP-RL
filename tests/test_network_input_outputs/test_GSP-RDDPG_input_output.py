# from gsp_rl.src.networks import (
#     RDDPGActorNetwork,
#     RDDPGCriticNetwork
# )

# import torch as T

# def test_building_actor_network():
#     id: int = 1
#     lr: float = 1e-4
#     input_size: int = 25
#     output_size: int = 1
#     fc1_dims:int  = 200
#     fc2_dims: int = 400
#     hidden_size = 4
#     meta_param_size = 2
#     num_layers = 1
#     batch_size = 16
#     RDDPG_Actor_Network = RDDPGActorNetwork(id, lr, input_size, output_size, hidden_size, meta_param_size, batch_size, num_layers, fc1_dims, fc2_dims)
#     for name, param in RDDPG_Actor_Network.named_parameters():
#         shape = param.shape
#         if name == 'ee.weight_ih_l0':
#             assert(shape[1] == input_size)
#         if name == 'meta_layer.weight':
#             assert(shape[0] == meta_param_size)
#         if name == 'fc1.weight':
#             assert(shape[1] == meta_param_size)
#             assert(shape[0] == fc1_dims)
#         if name == 'fc2.weight':
#             assert(shape[1] == fc1_dims)
#             assert(shape[0] == fc2_dims)
#         if name == 'fc3.weight':
#             assert(shape[1] == fc2_dims)
#             assert(shape[0] == output_size)

# def test_actor_forward():
#     id: int = 1
#     lr: float = 1e-4
#     input_size: int = 25
#     output_size: int = 1
#     fc1_dims:int  = 200
#     fc2_dims: int = 400
#     hidden_size = 4
#     meta_param_size = 2
#     num_layers = 1
#     batch_size = 16
#     RDDPG_Actor_Network = RDDPGActorNetwork(id, lr, input_size, output_size, hidden_size, meta_param_size, num_layers, batch_size, fc1_dims, fc2_dims)
#     random_observation = T.rand((batch_size, input_size))
#     #random_observation = T.rand((1, input_size))
#     assert(RDDPG_Actor_Network(random_observation).shape[1] == output_size)

# def test_building_critic_network():
#     id: int = 1
#     lr: float = 1e-4
#     input_size: int = 30
#     actor_output_size: int = 1
#     fc1_dims:int  = 200
#     fc2_dims: int = 400
#     DQN_Network = DDPGCriticNetwork(id, lr, input_size, actor_output_size, fc1_dims, fc2_dims)
#     for name, param in DQN_Network.named_parameters():
#         shape = param.shape
#         if name == 'fc1.weight':
#             assert(shape[1] == input_size+actor_output_size)
#             assert(shape[0] == fc1_dims)
#         elif name == 'fc2.weight':
#             assert(shape[1] == fc1_dims)
#             assert(shape[0] == fc2_dims)
#         elif name == 'fc3.weight':
#             assert(shape[1] == fc2_dims)
#             assert(shape[0] == 1)

# def test_critic_forward():
#     id: int = 1
#     lr: float = 1e-4
#     input_size: int = 30
#     actor_output_size: int = 1
#     fc1_dims:int  = 200
#     fc2_dims: int = 400
#     DDPG_Critic_Network = DDPGCriticNetwork(id, lr, input_size, actor_output_size, fc1_dims, fc2_dims)
#     random_input = T.rand((1, input_size))
#     random_action = T.rand((1, actor_output_size))
#     assert(DDPG_Critic_Network(random_input, random_action).shape[1] == 1)
    