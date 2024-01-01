import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

############################################################################
# Recurrent Layer for Environment Encoder
############################################################################
class EnvironmentEncoder(nn.Module):
    """
    LSTM Constructor for the network topology
    """
    def __init__(
            self,
            observation_size: int,
            hidden_size: int,
            meta_param_size: int,
            batch_size: int,
            num_layers: int,
            lr: float
    ) -> None:
        """
        Constructor
        """
        super(EnvironmentEncoder, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.observation = observation_size
        self.hidden_size = hidden_size
        self.meta_param_size = meta_param_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.ee = nn.LSTM(observation_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.meta_layer = nn.Linear(hidden_size, meta_param_size)

        self.ee_optimizer = optim.Adam(self.ee.parameters(), lr=lr, weight_decay= 1e-4)
        self.name = "Enviroment_Encoder"
        self.to(self.device)

    def forward(
            self,
            observation: T.Tensor,
            choose_action: bool = False
    ) -> None:
        """ Forward Propogation Step """
        hidden0 = (T.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device), T.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))
        lstm_out , (h_out, _) = self.ee(observation,hidden0)
        lstm_out = h_out.view(-1,self.hidden_size)
        meta_parameters = self.meta_layer(lstm_out)
        meta_parameters = T.relu(meta_parameters)
        if choose_action:
            meta_parameters = meta_parameters[-1]
        return meta_parameters

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