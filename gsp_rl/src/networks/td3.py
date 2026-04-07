"""Twin Delayed DDPG (TD3) Actor and Critic networks.

Same architecture family as DDPG but with required hidden layer dims and
separate learning rate params (alpha for actor, beta for critic). TD3 uses
two critic instances to reduce overestimation; the twin-critic logic is in
the learn method (NetworkAids.learn_TD3), not in the network classes.

See Also: docs/modules/networks.md
"""
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def fanin_init(size, fanin=None):
    """Fan-in weight initialization matching DDPG (Lillicrap et al. 2015)."""
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)


class TD3ActorNetwork(nn.Module):
    """Deterministic policy network for TD3.

    Same architecture as DDPGActorNetwork but uses default PyTorch weight
    initialization (no fanin_init) and requires fc1_dims/fc2_dims explicitly.
    Uses 'alpha' learning rate (vs DDPG's 'lr').

    Attributes:
        min_max_action: Action space bound for tanh scaling.
        name: Formatted as '{name}_{id}_TD3' for checkpoint files.
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

        self.init_weights(3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr = alpha, weight_decay = 1e-4)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def init_weights(self, init_w: float) -> None:
        """Initialize weights using fan-in init (matching DDPG)."""
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.mu.weight.data.uniform_(-init_w, init_w)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Compute deterministic action given state.

        Args:
            state: State tensor of shape (*, input_size).

        Returns:
            Action tensor of shape (*, output_size), bounded by min_max_action.
        """
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.mu(prob)
        mu = self.min_max_action * T.tanh(mu)

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
        if str(self.device) == 'cpu':
            self.load_state_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))

############################################################################
# Critic Network for TD3
############################################################################
class TD3CriticNetwork(nn.Module):
    """State-action value function (Q-network) for TD3.

    Same architecture as DDPGCriticNetwork. TD3 uses two instances
    (critic_1, critic_2) and takes the min for target computation.
    Concatenates state and action with dim=1 in forward().
    Uses 'beta' learning rate (vs DDPG's 'lr').

    Attributes:
        name: Formatted as '{name}_{id}_TD3' for checkpoint files.
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
        """Compute Q-value for a state-action pair.

        Args:
            state: State tensor of shape (batch, state_dim).
            action: Action tensor of shape (batch, action_dim).

        Returns:
            Q-value tensor of shape (batch, 1).
        """
        q1_action_value = F.relu(self.fc1(T.cat([state, action], dim = 1)))
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