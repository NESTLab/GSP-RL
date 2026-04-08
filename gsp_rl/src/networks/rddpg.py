"""Recurrent DDPG (RDDPG) network wrappers.

Composes EnvironmentEncoder (LSTM) with standard DDPG actor/critic networks.
The encoder transforms observation sequences into fixed-size encodings that
replace raw state as input to the DDPG networks.

Architecture:
    State -> EnvironmentEncoder -> encoding -> DDPGActorNetwork -> Action
    State -> EnvironmentEncoder -> encoding -> DDPGCriticNetwork(encoding, action) -> Q-value

In make_RDDPG_networks: actor and critic share one EnvironmentEncoder
instance (shared_ee), while target networks get separate encoder instances
for proper gradient flow isolation.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RDDPGActorNetwork(nn.Module):
    """RDDPG actor: EnvironmentEncoder + DDPGActorNetwork composition.

    Owns an Adam optimizer over all parameters (encoder + DDPG actor).
    Device is inherited from the encoder.

    Attributes:
        ee: EnvironmentEncoder (LSTM-based).
        actor: DDPGActorNetwork.
    """
    def __init__(self, environmental_encoder, ddpg_actor):
        super().__init__()
        self.ee = environmental_encoder
        self.actor = ddpg_actor
        # Use encoder's device (CPU on MPS due to LSTM fallback, GPU on CUDA)
        self.device = self.ee.device
        self.actor.to(self.device)
        self.actor.device = self.device
        self.optimizer = optim.Adam(self.parameters(), lr = ddpg_actor.lr, weight_decay = 1e-4)

    def forward(self, x, hidden=None):
        """Encode observation through LSTM, then compute action via DDPG actor.

        Args:
            x: Observation tensor of shape (seq_len, input_size) or
               (batch, seq_len, input_size).
            hidden: Optional (h_0, c_0) tuple for the LSTM encoder.

        Returns:
            Tuple of (mu, (h_n, c_n)):
                mu: Action tensor.
                (h_n, c_n): Final LSTM hidden state.
        """
        encoding, hidden_out = self.ee(x, hidden=hidden)
        mu = self.actor(encoding)
        return mu, hidden_out
    
    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.save_checkpoint(path, intention)
        self.actor.save_checkpoint(path, intention)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.load_checkpoint(path, intention)
        self.actor.load_checkpoint(path, intention)


class RDDPGCriticNetwork(nn.Module):
    """RDDPG critic: EnvironmentEncoder + DDPGCriticNetwork composition.

    Owns an Adam optimizer over all parameters (encoder + DDPG critic).
    Note: save_checkpoint does NOT save the encoder (assumed saved by actor).
    load_checkpoint DOES load the encoder.

    Attributes:
        ee: EnvironmentEncoder (LSTM-based).
        critic: DDPGCriticNetwork.
    """
    def __init__(self, environmental_encoder, ddpg_critic):
        super().__init__()
        self.ee = environmental_encoder
        self.critic = ddpg_critic
        # Use encoder's device (CPU on MPS due to LSTM fallback, GPU on CUDA)
        self.device = self.ee.device
        self.critic.to(self.device)
        self.critic.device = self.device
        self.optimizer = optim.Adam(self.parameters(), lr = ddpg_critic.lr, weight_decay = 1e-4)
    
    def forward(self, state, action, hidden=None):
        """Encode state through LSTM, then compute Q-value via DDPG critic.

        Args:
            state: Observation tensor of shape (seq_len, input_size) or
                   (batch, seq_len, input_size).
            action: Action tensor of shape (seq_len, action_dim).
            hidden: Optional (h_0, c_0) tuple for the LSTM encoder.

        Returns:
            Tuple of (action_value, (h_n, c_n)):
                action_value: Q-value tensor.
                (h_n, c_n): Final LSTM hidden state.
        """
        encoding, hidden_out = self.ee(state, hidden=hidden)
        action_value = self.critic(encoding, action)
        return action_value, hidden_out
    
    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        # NOTE The ee will be saved in the actor network
        # self.ee.save_checkpoint(path, intention)
        self.critic.save_checkpoint(path, intention)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.load_checkpoint(path, intention)
        self.critic.load_checkpoint(path, intention)

