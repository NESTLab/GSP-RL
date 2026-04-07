"""DDPG Actor and Critic networks for continuous action spaces.

Provides DDPGActorNetwork (deterministic policy) and DDPGCriticNetwork
(state-action value function). Both use fan-in weight initialization.
Also serves as the base architecture composed by RDDPG wrappers in rddpg.py.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


def fanin_init(size, fanin=None):
    """Initialize weights using fan-in uniform distribution.

    Args:
        size: Shape tuple for the weight tensor.
        fanin: Fan-in value. Defaults to size[0] (input features).

    Returns:
        Tensor of given size with values uniform in [-1/sqrt(fanin), 1/sqrt(fanin)].
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)


class DDPGActorNetwork(nn.Module):
    """Deterministic policy network for DDPG.

    Architecture: Linear(input_size, 400) -> ReLU -> Linear(400, 300) -> ReLU
    -> Linear(300, output_size) -> Tanh * min_max_action.

    Output is bounded to [-min_max_action, min_max_action] via tanh scaling.
    Used directly for DDPG and as a component inside RDDPGActorNetwork.

    Attributes:
        min_max_action: Action space bound for tanh scaling.
        name: Formatted as '{name}_{id}_DDPG' for checkpoint files.
    """
    def __init__(
            self,
            id: int,
            lr: float,
            input_size: int,
            output_size: int,
            fc1_dims: int = 400,
            fc2_dims: int = 300,
            name: str = "DDPG_Actor",
            min_max_action: float = 1.0
    ) -> None:
        """Initialize DDPG actor network.

        Args:
            id: Agent identifier, embedded in checkpoint name.
            lr: Learning rate for Adam optimizer.
            input_size: Observation dimensionality (or encoding size for RDDPG).
            output_size: Action space dimensionality.
            fc1_dims: First hidden layer width.
            fc2_dims: Second hidden layer width.
            name: Base name for checkpoint files.
            min_max_action: Tanh output scaling factor.
        """
        super().__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.min_max_action = min_max_action
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(input_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, output_size)

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
        """Compute deterministic action given state.

        Args:
            x: State tensor of shape (*, input_size).

        Returns:
            Action tensor of shape (*, output_size), bounded by min_max_action.
        """
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
    """State-action value function (Q-network) for DDPG.

    Architecture: Linear(input_size, 400) -> ReLU -> Linear(400, 300) -> ReLU
    -> Linear(300, 1). Concatenates state and action internally in forward().

    IMPORTANT: input_size must equal state_dim + action_dim because forward()
    calls T.cat([state, action], dim=-1) before the first layer.

    Attributes:
        name: Formatted as '{name}_{id}_DDPG' for checkpoint files.
    """
    def __init__(self,
                 id: int,
                 lr: float,
                 input_size: int,
                 output_size: int,
                 fc1_dims: int = 400,
                 fc2_dims: int = 300,
                 name: str = "DDPG_Critic"
    ):
        """Initialize DDPG critic network.

        Args:
            id: Agent identifier, embedded in checkpoint name.
            lr: Learning rate for Adam optimizer.
            input_size: Must be state_dim + action_dim (state-action concat).
            output_size: Kept for interface consistency (output is always 1).
            fc1_dims: First hidden layer width.
            fc2_dims: Second hidden layer width.
            name: Base name for checkpoint files.
        """
        super().__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = nn.Linear(input_size, self.fc1_dims)
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

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        """Compute Q-value for a state-action pair.

        Args:
            state: State tensor of shape (batch, state_dim).
            action: Action tensor of shape (batch, action_dim).

        Returns:
            Q-value tensor of shape (batch, 1).
        """
        action_value = self.fc1(T.cat([state, action], dim = -1))
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
        if str(self.device) == 'cpu':
            self.load_state_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))