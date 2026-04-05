"""LSTM-based Environment Encoder for recurrent RL variants.

Provides the EnvironmentEncoder which maps observation sequences into a
fixed-size encoding via: Linear(input_size, embedding_size) -> LSTM ->
Linear(hidden_size, output_size). Used as a component in RDDPG networks.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.optim as optim


class EnvironmentEncoder(nn.Module):
    """LSTM encoder that transforms observation sequences into fixed encodings.

    Architecture: Linear embedding -> LSTM (multi-layer) -> Linear projection.
    Composed into RDDPGActorNetwork and RDDPGCriticNetwork. The actor and
    critic share one encoder instance; target networks get separate instances.

    Note: No optimizer is defined here -- the RDDPG wrapper creates an Adam
    optimizer over all parameters (encoder + DDPG network).

    Attributes:
        embedding: Linear(input_size, embedding_size).
        ee: LSTM(embedding_size, hidden_size, num_layers, batch_first=True).
        meta_layer: Linear(hidden_size, output_size).
        name: "Enviroment_Encoder" (historical typo preserved).
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
        """Initialize EnvironmentEncoder.

        Args:
            input_size: Raw observation dimensionality.
            output_size: Encoding dimensionality (meta_param_size).
            hidden_size: LSTM hidden state size.
            embedding_size: Linear embedding layer output size.
            batch_size: Stored but not used internally.
            num_layers: Number of stacked LSTM layers.
            lr: Stored but optimizer is created in RDDPG wrapper.
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
    ) -> T.Tensor:
        """Encode an observation (or sequence) through embedding + LSTM + projection.

        Args:
            observation: Tensor of shape (seq_len, input_size) or (batch, input_size).

        Returns:
            Encoding tensor of shape (seq_len, 1, output_size). The middle dim=1
            comes from the view reshape before LSTM.
        """
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