import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

############################################################################
# Action Network for DQN
############################################################################
class DQN(nn.Module):
    """
    DQN Constructor for the network topology
    """
    def __init__(
            self,
            id: int,
            lr: float,
            num_actions: int,
            observation_size: int,
            num_ops_per_action: int,
            fc1_dims: int = 64,
            fc2_dims: int = 128,
            name: str = 'DQN'
    ) -> None:
        """
        constructor 
        """
        super().__init__()

        self.name = name

        output_dims = num_ops_per_action**num_actions

        self.fc1 = nn.Linear(observation_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """
        forward propogation of state through the network
        """
        x = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x))
        actions = self.fc3(x1)

        return actions

    def save_checkpoint(self, path: str, intention: bool = False):
        """ Saves the model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention: bool = False):
        """ Loads the model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))