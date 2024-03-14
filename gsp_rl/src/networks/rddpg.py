import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
"""
Note that the EE for the actor and the critic are the same network

                            State, Action
                              |      |
                              EE     |
                              /\     |
                             /  \    |
                            /    \   |
                           /      \  |
                        Actor    Critic
                          |         |
                        Action    Value
"""

class RDDPGActorNetwork(nn.Module):
    def __init__(self, environmental_encoder, ddpg_actor):
        super().__init__()
        self.ee = environmental_encoder
        self.actor = ddpg_actor
        self.device = self.ee.device
        self.optimizer = optim.Adam(self.parameters(), lr = ddpg_actor.lr, weight_decay = 1e-4)

    def forward(self, x: T.Tensor) -> T.Tensor:
        encoding = self.ee(x)
        mu = self.actor(encoding)
        return mu
    
    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.save_checkpoint(path, intention)
        self.actor.save_checkpoint(path, intention)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.load_checkpoint(path, intention)
        self.actor.load_checkpoint(path, intention)


class RDDPGCriticNetwork(nn.Module):
    def __init__(self, environmental_encoder, ddpg_critic):
        super().__init__()
        self.ee = environmental_encoder
        self.critic = ddpg_critic
        self.device = self.ee.device
        self.optimizer = optim.Adam(self.parameters(), lr = ddpg_critic.lr, weight_decay = 1e-4)
    
    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        encoding = self.ee(state)
        action_value = self.critic(encoding, action)
        return action_value
    
    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        # NOTE The ee will be saved in the actor network
        # self.ee.save_checkpoint(path, intention)
        self.critic.save_checkpoint(path, intention)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.load_checkpoint(path, intention)
        self.actor.load_checkpoint(path, intention)

