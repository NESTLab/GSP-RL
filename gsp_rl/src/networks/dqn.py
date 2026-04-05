"""Deep Q-Network (DQN) for discrete action spaces.

Provides a simple feedforward Q-network: Linear(input_size, 64) -> ReLU ->
Linear(64, 128) -> ReLU -> Linear(128, output_size). Outputs raw Q-values
(no output activation). Used in pairs (q_eval, q_next) by NetworkAids.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    """Feedforward Q-network for DQN and value-based RL.

    Produces Q-values for each discrete action given a state observation.
    The network dict stores two instances: 'q_eval' (online) and 'q_next' (target).

    Attributes:
        fc1: First hidden layer (input_size -> fc1_dims).
        fc2: Second hidden layer (fc1_dims -> fc2_dims).
        fc3: Output layer (fc2_dims -> output_size), produces Q-values.
        optimizer: Adam with weight_decay=1e-4.
        loss: MSELoss instance for TD error computation.
        device: Auto-detected cuda:0 or cpu.
    """
    def __init__(
            self,
            id: int,
            lr: float,
            input_size: int,
            output_size: int,
            fc1_dims: int = 64,
            fc2_dims: int = 128,
            name: str = 'DQN'
    ) -> None:
        """Initialize DQN network.

        Args:
            id: Agent identifier (unused in network, kept for interface consistency).
            lr: Learning rate for Adam optimizer.
            input_size: Observation space dimensionality.
            output_size: Number of discrete actions (Q-value per action).
            fc1_dims: First hidden layer width.
            fc2_dims: Second hidden layer width.
            name: Network name for checkpoint file naming.
        """
        super().__init__()

        self.name = name

        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Compute Q-values for all actions given a state.

        Args:
            state: Observation tensor of shape (*, input_size).

        Returns:
            Q-values tensor of shape (*, output_size).
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
        if self.device == 'cpu':
            self.load_stat_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))