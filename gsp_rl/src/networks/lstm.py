import torch as T
import torch.nn as nn
import torch.optim as optim

############################################################################
# Recurrent Layer for Environment Encoder
############################################################################
class EnvironmentEncoder(nn.Module):
    """
    LSTM Constructor for the network topology
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size: int,
            embedding_size: int,
            batch_size: int,
            num_layers: int,
            lr: float
    ) -> None:
        """
        Constructor
        """
        super().__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(self.input_size, self.embedding_size)
        self.ee = nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            num_layers = self.num_layers,
            batch_first=True
        )
        self.meta_layer = nn.Linear(self.hidden_size, self.output_size)

        #self.ee_optimizer = optim.Adam(self.ee.parameters(), lr=lr, weight_decay= 1e-4)
        self.name = "Enviroment_Encoder"
        self.to(self.device)

    def forward(
            self,
            observation: T.Tensor,
    ) -> None:
        """ Forward Propogation Step """
        embed = self.embedding(observation)
        lstm_out, _ = self.ee(embed.view(embed.shape[0], 1, -1))
        out = self.meta_layer(lstm_out)
        return out

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
        if self.device == 'cpu':
            self.load_stat_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))