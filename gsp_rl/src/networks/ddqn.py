"""Double Deep Q-Network (DDQN) for discrete action spaces.

Same architecture as DQN. The double-Q trick is implemented in the learn
method (NetworkAids.learn_DDQN), not in the network itself: q_eval selects
actions, q_next evaluates them, reducing overestimation bias.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDQN(nn.Module):
    """Feedforward Q-network for Double DQN.

    Architecturally identical to DQN. The double-Q decoupling happens in
    the learn loop, not in the network structure.

    Attributes:
        fc1: First hidden layer (input_size -> fc1_dims).
        fc2: Second hidden layer (fc1_dims -> fc2_dims).
        fc3: Output layer (fc2_dims -> output_size), produces Q-values.
        optimizer: Adam with weight_decay=1e-4.
        loss: MSELoss instance.
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
            name: str = 'DDQN'
    ) -> None:
        """Initialize DDQN network.

        Args:
            id: Agent identifier.
            lr: Learning rate for Adam optimizer.
            input_size: Observation space dimensionality.
            output_size: Number of discrete actions.
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

    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Saves the model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Loads the model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        if str(self.device) == 'cpu':
            self.load_state_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))