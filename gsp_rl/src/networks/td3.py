import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

############################################################################
# Actor Network for TD3
############################################################################
class TD3ActorNetwork(nn.Module):
    """
    TD3 Actor Constructor for the network topology
    """
    def __init__(
            self,
            id: int,
            alpha: float,
            input_size: int,
            output_size: int,
            fc1_dims: int,
            fc2_dims: int,
            name: str = "TD3_Actor",
            min_max_action: int = 1
    ) -> None:
        """ Constructor """
        super().__init__()
        self.input_dims = input_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_size
        self.min_max_action = min_max_action
        self.name = name +'_'+str(id)+'_TD3'


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha, weight_decay = 1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """
        Forward Propogation Step
        """
        print('[TD3] Min Max', self.min_max_action)
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.min_max_action * T.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Save Model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Load Model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))

############################################################################
# Critic Network for TD3
############################################################################
class TD3CriticNetwork(nn.Module):
    """
    TD3 Critic Constructor for the network topology
    """
    def __init__(
            self,
            id: int,
            beta: float,
            input_size: int,
            output_size: int,
            fc1_dims: int,
            fc2_dims: int,
            name: str = "TD3_Critic"
    ) -> None:
        """ Constructor """
        super().__init__()
        self.input_dims = input_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_size
        self.name = name +'_'+str(id)+'_TD3'

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta, weight_decay = 1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        """
        Forward Propogation Step
        """
        q1_action_value = F.relu(self.fc1(T.cat([state, action[:,:2]], dim = 1))) #Remove [:,:2] from actions if grippers action needed as input
        q1_action_value = F.relu(self.fc2(q1_action_value))
        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Save Model"""
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Load Model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))