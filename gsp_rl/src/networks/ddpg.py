import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

def fanin_init(size, fanin = None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)

############################################################################
# Actor Network for DDPG
############################################################################
class DDPGActorNetwork(nn.Module):
    """
    DDPG Actor Constructor for the network topology
    """
    def __init__(
            self,
            id: int,
            num_actions: int,
            observation_size: int,
            lr: float,
            name: str = "DDPG_Actor",
            min_max_action: float = 1.0
    ) -> None:
        """
        constructor 
        """
        super().__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        output_dims = num_actions

        self.min_max_action = min_max_action

        self.fc1_dims = 400
        self.fc2_dims = 300

        self.fc1 = nn.Linear(observation_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, output_dims)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.init_weights(3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.name = name+'_'+str(id)+'_DDPG'

        self.to(self.device)


    def init_weights(self, init_w: float) -> None:
        """ Initializes weights of the network"""
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.mu.weight.data.uniform_(-init_w, init_w)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """ Forward Propogation Steo"""
        prob = self.fc1(x)
        prob = self.relu(prob)
        prob = self.fc2(prob)
        prob = self.relu(prob)
        mu = self.mu(prob)
        mu = self.min_max_action*self.tanh(mu)
        return mu

    def save_checkpoint(self, path: str, intention=False) -> None:
        """ Saves the Model"""
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention=False) -> None:
        """ Saves the Model"""
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))


############################################################################
# Critic Network for DDPG
############################################################################
class DDPGCriticNetwork(nn.Module):
    """
    DDPG Critic Constructor for the network topology
    """
    def __init__(self,
                 id: int,
                 num_actions: int,
                 observation_size: int,
                 lr: float,
                 name: str = "DDPG_Critic"
    ):
        """
        Constuctor
        """
        super().__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.fc1_dims = 400
        self.fc2_dims = 300
        self.fc1 = nn.Linear(observation_size+num_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.relu = nn.ReLU()
        self.init_weights(3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.name = name+'_'+str(id)+'_DDPG'
        self.to(self.device)

    def init_weights(self, init_w: float) -> None:
        """
        Initializes the weights of the network
        """
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.q.weight.data.uniform_(-init_w, init_w)

    def forward(self, X: T.Tensor) -> T.Tensor:
        """
        Forward Propogation Step"""
        state, action = X
        action_value = self.fc1(T.cat([state, action], 1))
        action_value = self.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = self.relu(action_value)
        action_value = self.q(action_value)
        return action_value

    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Saves Model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Loads Model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))