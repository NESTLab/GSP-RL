# import torch as T
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# def fanin_init(size, fanin = None):
#     fanin = fanin or size[0]
#     v = 1. / np.sqrt(fanin)
#     return T.Tensor(size).uniform_(-v, v)

# ############################################################################
# # Actor Network for DDPG
# ############################################################################
# class RDDPGActorNetwork(nn.Module):
#     """
#     DDPG Actor Constructor for the network topology
#     """
#     def __init__(
#             self,
#             id: int,
#             lr: float,
#             input_size: int,
#             output_size: int,
#             lstm_hidden_size: int, 
#             lstm_meta_param_size: int,
#             lstm_num_layers: int,
#             batch_size: int,
#             fc1_dims: int = 400,
#             fc2_dims: int = 300,
#             name: str = "DDPG_Actor",
#             min_max_action: float = 1.0
#     ) -> None:
#         """
#         constructor 
#         """
#         super().__init__()

#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

#         self.input_size = input_size
#         self.lstm_hidden_size = lstm_hidden_size
#         self.lstm_meta_param_size = lstm_meta_param_size
#         self.batch_size = batch_size
#         self.lstm_num_layers = lstm_num_layers
#         self.min_max_action = min_max_action

#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims

#         self.ee = nn.LSTM(input_size, lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
#         self.meta_layer = nn.Linear(lstm_hidden_size, lstm_meta_param_size)
#         self.fc1 = nn.Linear(lstm_meta_param_size, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.mu = nn.Linear(self.fc2_dims, output_size)

#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()

#         self.init_weights(3e-3)

#         self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

#         self.name = name+'_'+str(id)+'_RDDPG'

#         self.to(self.device)


#     def init_weights(self, init_w: float) -> None:
#         """ Initializes weights of the network"""
#         #self.ee.weigh.data = fanin_init(self.ee.weight.data.size())
#         self.meta_layer.weight.data = fanin_init(self.meta_layer.weight.data.size())
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
#         self.mu.weight.data.uniform_(-init_w, init_w)

#     def forward(self, x: T.Tensor) -> T.Tensor:
#         """ Forward Propogation Step"""
#         print(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size)
#         hidden0 = (
#             T.zeros(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size).to(self.device), 
#             T.zeros(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size).to(self.device)
#         )
#         lstm_out , (h_out, _) = self.ee(x,hidden0)
#         lstm_out = h_out.view(-1,self.hidden_size)
#         meta_parameters = self.meta_layer(lstm_out)
#         meta_parameters = T.relu(meta_parameters)
#         prob = self.fc1(meta_parameters)
#         prob = self.relu(prob)
#         prob = self.fc2(prob)
#         prob = self.relu(prob)
#         mu = self.mu(prob)
#         mu = self.min_max_action*self.tanh(mu)
#         return mu

#     def save_checkpoint(self, path: str, intention=False) -> None:
#         """ Saves the Model"""
#         network_name = self.name
#         if intention:
#             network_name += "_intention"
#         print('... saving', network_name,'...')
#         T.save(self.state_dict(), path + '_' + network_name)

#     def load_checkpoint(self, path: str, intention=False) -> None:
#         """ Saves the Model"""
#         network_name = self.name
#         if intention:
#             network_name += "_intention"
#         print('... loading', network_name, '...')
#         self.load_state_dict(T.load(path + '_' + network_name))


# ############################################################################
# # Critic Network for DDPG
# ############################################################################
# class RDDPGCriticNetwork(nn.Module):
#     """
#     DDPG Critic Constructor for the network topology
#     """
#     def __init__(
#             self,
#             id: int,
#             lr: float,
#             input_size: int,
#             actor_output_size: int,
#             lstm_hidden_size: int, 
#             lstm_meta_param_size: int,
#             lstm_num_layers: int,
#             batch_size: int,
#             fc1_dims: int = 400,
#             fc2_dims: int = 300,
#             name: str = "DDPG_Critic"
#     ):
#         """
#         - input_size: This should match the input size to your actor network
#         - actor_ouput_size: This should be the same as the output_size of your actor network
#         """
#         super().__init__()

#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims

#         self.lstm_hidden_size = lstm_hidden_size
#         self.lstm_meta_param_size = lstm_meta_param_size
#         self.batch_size = batch_size,
#         self.lstm_num_layers = lstm_num_layers

#         self.ee = nn.LSTM(input_size + actor_output_size, lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
#         self.meta_layer = nn.Linear(lstm_hidden_size, lstm_meta_param_size)
#         self.fc1 = nn.Linear(lstm_meta_param_size, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.q = nn.Linear(self.fc2_dims, 1)

#         self.relu = nn.ReLU()
#         self.init_weights(3e-3)

#         self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

#         self.name = name+'_'+str(id)+'_DDPG'
#         self.to(self.device)

#     def init_weights(self, init_w: float) -> None:
#         """
#         Initializes the weights of the network
#         """
#         #self.ee.weigh.data = fanin_init(self.ee.weight.data.size())
#         self.meta_layer.weight.data = fanin_init(self.meta_layer.weight.data.size())
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
#         self.q.weight.data.uniform_(-init_w, init_w)

#     def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
#         """
#         Forward Propogation Step"""
#         hidden0 = (
#             T.zeros(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size).to(self.device),
#             T.zeros(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size).to(self.device)
#         )
#         lstm_out , (h_out, _) = self.ee(T.cat([state, action], 1), hidden0)
#         lstm_out = h_out.view(-1,self.lstm_hidden_size)
#         meta_parameters = self.meta_layer(lstm_out)
#         meta_parameters = T.relu(meta_parameters)
#         action_value = self.fc1(meta_parameters)
#         action_value = self.relu(action_value)
#         action_value = self.fc2(action_value)
#         action_value = self.relu(action_value)
#         action_value = self.q(action_value)
#         return action_value

#     def save_checkpoint(self, path: str, intention: bool = False) -> None:
#         """ Saves Model """
#         network_name = self.name
#         if intention:
#             network_name += "_intention"
#         print('... saving', network_name,'...')
#         T.save(self.state_dict(), path + '_' + network_name)

#     def load_checkpoint(self, path: str, intention: bool = False) -> None:
#         """ Loads Model """
#         network_name = self.name
#         if intention:
#             network_name += "_intention"
#         print('... loading', network_name, '...')
#         self.load_state_dict(T.load(path + '_' + network_name))