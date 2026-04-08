"""LSTM-based Environment Encoder for recurrent RL variants.

Provides the EnvironmentEncoder which maps observation sequences into a
fixed-size encoding via: Linear(input_size, embedding_size) -> LSTM ->
Linear(hidden_size, output_size). Used as a component in RDDPG networks.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.optim as optim
from gsp_rl.src.networks import get_device


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
        self.device = get_device(recurrent=True)

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

    def forward(self, observation, hidden=None):
        """Encode observation through embedding + LSTM + projection.

        Args:
            observation: Tensor of shape (seq_len, input_size) for single sample,
                         or (batch, seq_len, input_size) for batched input.
            hidden: Optional (h_0, c_0) tuple. If None, LSTM uses zeros.
                    Shape: (num_layers, batch, hidden_size) for each.

        Returns:
            Tuple of (output, (h_n, c_n)):
                output: Shape (seq_len, output_size) for single,
                        or (batch, seq_len, output_size) for batched.
                (h_n, c_n): Final hidden state.
        """
        # Handle single vs batched input
        if observation.dim() == 2:
            # Single sample: (seq_len, input) -> add batch dim
            observation = observation.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # observation is now (batch, seq_len, input_size)
        embed = self.embedding(observation)  # (batch, seq_len, embed_size)

        if hidden is not None:
            lstm_out, (h_n, c_n) = self.ee(embed, hidden)
        else:
            lstm_out, (h_n, c_n) = self.ee(embed)

        out = self.meta_layer(lstm_out)  # (batch, seq_len, output_size)

        if squeeze_batch:
            out = out.squeeze(0)  # back to (seq_len, output_size)

        return out, (h_n, c_n)

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
        if str(self.device) == 'cpu':
            self.load_state_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))