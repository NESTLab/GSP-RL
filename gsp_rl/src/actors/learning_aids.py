"""Network factory methods, learning algorithms, and hyperparameter management.

Contains the three-level class hierarchy base:
- Hyperparameters: Loads and stores all config values from YAML dict.
- NetworkAids(Hyperparameters): Factory methods (make_*_networks), learning
  algorithms (learn_*), action selection (*_choose_action), and memory
  management (sample_memory, store_transition).

Actor (in actor.py) inherits from NetworkAids, completing the chain:
Actor -> NetworkAids -> Hyperparameters.

See Also: docs/modules/actors.md, docs/algorithms.md
"""
from gsp_rl.src.networks import (
    DQN,
    DDQN,
    DDPGActorNetwork,
    DDPGCriticNetwork,
    RDDPGActorNetwork,
    RDDPGCriticNetwork,
    TD3ActorNetwork,
    TD3CriticNetwork,
    EnvironmentEncoder,
    AttentionEncoder
)
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam

import numpy as np
import logging

_learn_logger = logging.getLogger("stelaris.learn")


def _check_nan(value, name):
    """Raise RuntimeError if value is NaN or Inf. Works with floats and tensors."""
    if isinstance(value, T.Tensor):
        if T.isnan(value).any() or T.isinf(value).any():
            raise RuntimeError(f"NaN detected in {name}: {value}")
    else:
        if not np.isfinite(value):
            raise RuntimeError(f"NaN detected in {name}: {value}")


Loss = nn.MSELoss()


class Hyperparameters:
    """Configuration container loaded from a YAML config dict.

    Maps YAML keys to instance attributes. Notable name mappings:
    - GSP_LEARNING_FREQUENCY -> self.gsp_learning_offset
    - REPLACE_TARGET_COUNTER -> self.replace_target_ctr

    Also initializes self.time_step = 0 (used by TD3 warmup).
    """
    def __init__(self, config):
        self.gamma = config['GAMMA']
        self.tau = config['TAU']
        self.alpha = config['ALPHA']
        self.beta = config['BETA']
        self.lr = config['LR']

        self.epsilon = config['EPSILON']
        self.eps_min = config['EPS_MIN']
        self.eps_dec = config['EPS_DEC']

        self.gsp_learning_offset = config['GSP_LEARNING_FREQUENCY'] #learn after every 1000 action network learning steps
        self.gsp_batch_size = config['GSP_BATCH_SIZE']

        self.batch_size = config['BATCH_SIZE']
        self.mem_size = config['MEM_SIZE']
        self.replace_target_ctr = config['REPLACE_TARGET_COUNTER']

        self.noise = config['NOISE']
        self.update_actor_iter = config['UPDATE_ACTOR_ITER']
        self.warmup = config['WARMUP']
        self.time_step = 0

class NetworkAids(Hyperparameters):
    """Network factory, learning algorithms, action selection, and memory management.

    All methods operate on a 'networks' dict (plain dict, not a class) that
    contains the neural networks, replay buffer, learning scheme string, and
    step counter. This dict is either self.networks or self.gsp_networks,
    passed explicitly to allow the same learn/action methods to serve both
    the main action network and the GSP prediction network.
    """
    def __init__(self, config):
        super().__init__(config)
    def make_DQN_networks(self, nn_args):
        return {'q_eval':DQN(**nn_args), 'q_next':DQN(**nn_args)}
    
    def make_DDQN_networks(self, nn_args):
        return {'q_eval':DDQN(**nn_args), 'q_next':DDQN(**nn_args)}
    
    def make_DDPG_networks(self, actor_nn_args, critic_nn_args):
        DDPG_networks = {
                        'actor': DDPGActorNetwork(**actor_nn_args, name = 'actor'),
                        'target_actor': DDPGActorNetwork(**actor_nn_args, name = 'target_actor'),
                        'critic': DDPGCriticNetwork(**critic_nn_args, name = 'critic_1'),
                        'target_critic': DDPGCriticNetwork(**critic_nn_args, name = 'target_critic_1')}
        return DDPG_networks

    def make_TD3_networks(self, actor_nn_args, critic_nn_args):
        TD3_networks = {
                        'actor': TD3ActorNetwork(**actor_nn_args, name = 'actor'),
                        'target_actor': TD3ActorNetwork(**actor_nn_args, name = 'target_actor'),
                        'critic_1': TD3CriticNetwork(**critic_nn_args, name = 'critic_1'),
                        'target_critic_1': TD3CriticNetwork(**critic_nn_args, name = 'target_critic_1'),
                        'critic_2': TD3CriticNetwork(**critic_nn_args, name = 'critic_2'),
                        'target_critic_2': TD3CriticNetwork(**critic_nn_args, name = 'target_critic_2')}
        return TD3_networks
    
    def make_RDDPG_networks(self, lstm_nn_args, actor_nn_args, critic_nn_args):
        shared_ee = EnvironmentEncoder(**lstm_nn_args)
        RDDPG_networks = {
            'actor': RDDPGActorNetwork(shared_ee, DDPGActorNetwork(**actor_nn_args, name='actor')),
            'target_actor': RDDPGActorNetwork(EnvironmentEncoder(**lstm_nn_args), DDPGActorNetwork(**actor_nn_args, name='target_actor')),
            'critic': RDDPGCriticNetwork(shared_ee, DDPGCriticNetwork(**critic_nn_args, name = 'critic')),
            'target_critic':RDDPGCriticNetwork(EnvironmentEncoder(**lstm_nn_args), DDPGCriticNetwork(**critic_nn_args, name = 'target_critic'))
        }
        return RDDPG_networks
    
    def make_Environmental_Encoder(self, nn_args):
        lstm_networks = {'ee': EnvironmentEncoder(**nn_args)}
        return lstm_networks

    def make_Attention_Encoder(self, nn_args):
        Attention_networks = {'attention': AttentionEncoder(**nn_args)}
        return Attention_networks

    def update_DDPG_network_parameters(self, tau, networks):
        # Update Actor Network
        for target_param, param in zip(networks['target_actor'].parameters(), networks['actor'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        # Update Critic Network
        for target_param, param in zip(networks['target_critic'].parameters(), networks['critic'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        return networks

    def update_TD3_network_parameters(self, tau, networks):
        actor_params = networks['actor'].named_parameters()
        critic_1_params = networks['critic_1'].named_parameters()
        critic_2_params = networks['critic_2'].named_parameters()
        target_actor_params = networks['target_actor'].named_parameters()
        target_critic_1_params = networks['target_critic_1'].named_parameters()
        target_critic_2_params = networks['target_critic_2'].named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()

        networks['target_critic_1'].load_state_dict(critic_1)
        networks['target_critic_2'].load_state_dict(critic_2)
        networks['target_actor'].load_state_dict(actor)

        return networks

    def DQN_DDQN_choose_action(self, observation, networks):
        state = T.tensor(observation, dtype = T.float).to(networks['q_eval'].device)
        action_values = networks['q_eval'].forward(state)
        return T.argmax(action_values).item()

    def DQN_DDQN_choose_action_batch(self, observations, networks):
        """Batched action selection for DQN/DDQN. Returns list of action indices."""
        states = T.tensor(np.array(observations), dtype=T.float).to(networks['q_eval'].device)
        action_values = networks['q_eval'].forward(states)
        return T.argmax(action_values, dim=1).cpu().tolist()
    
    def DDPG_choose_action(self, observation, networks):
        if networks['learning_scheme'] == 'RDDPG':
            # if using LSTM we need to add an extra dimension
            state = T.tensor(np.array(observation), dtype=T.float).to(networks['actor'].device)
            mu, _ = networks['actor'].forward(state)
            return mu.unsqueeze(0)
        else:
            state = T.tensor(observation, dtype = T.float).to(networks['actor'].device)
            return networks['actor'].forward(state).unsqueeze(0)
        
    
    def DDPG_choose_action_batch(self, observations, networks):
        """Batched action selection for DDPG. Returns (batch, output_size) numpy array."""
        if networks['learning_scheme'] == 'RDDPG':
            # RDDPG uses sequences — cannot batch across robots (stateful LSTM)
            raise NotImplementedError("RDDPG cannot be batched — use sequential choose_action")
        states = T.tensor(np.array(observations), dtype=T.float).to(networks['actor'].device)
        return networks['actor'].forward(states)

    def TD3_choose_action_batch(self, observations, networks, n_actions):
        """Batched action selection for TD3. Returns list of (1, output_size) numpy arrays."""
        if self.time_step < self.warmup:
            batch_size = len(observations)
            mus = T.tensor(np.random.normal(scale=self.noise, size=(batch_size, n_actions)),
                           dtype=T.float).to(networks['actor'].device)
        else:
            states = T.tensor(np.array(observations), dtype=T.float).to(networks['actor'].device)
            mus = networks['actor'].forward(states).to(networks['actor'].device)
        noise = T.tensor(np.random.normal(scale=self.noise, size=mus.shape),
                         dtype=T.float).to(networks['actor'].device)
        mus_prime = T.clamp(mus + noise, -networks['actor'].min_max_action,
                            networks['actor'].min_max_action)
        self.time_step += 1
        return mus_prime.cpu().detach().numpy()

    def TD3_choose_action(self, observation, networks, n_actions):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale = self.noise,
                                           size = (n_actions,)),
                          dtype=T.float).to(networks['actor'].device)
        else:
            state = T.tensor(observation, dtype = T.float).to(networks['actor'].device)
            mu = networks['actor'].forward(state).to(networks['actor'].device)
        mu_prime = mu + T.tensor(np.random.normal(scale = self.noise), dtype = T.float).to(networks['actor'].device)
        mu_prime = T.clamp(mu_prime, -networks['actor'].min_max_action, networks['actor'].min_max_action)
        self.time_step += 1
        return mu_prime.unsqueeze(0).cpu().detach().numpy()
    
    def Attention_choose_action(self, observation, networks):
        return networks['attention'](observation).cpu().detach().numpy()

    
    def learn_DQN(self, networks):
        networks['q_eval'].optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory(networks)

        indices = T.LongTensor(np.arange(self.batch_size).astype(np.int64))

        q_pred = networks['q_eval'](states)[indices, actions.type(T.LongTensor)]

        q_next = networks['q_next'](states_).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next

        loss = networks['q_eval'].loss(q_target, q_pred).to(networks['q_eval'].device)
        loss.backward()
        _check_nan(loss, f"DQN loss at step {networks['learn_step_counter']}")

        networks['q_eval'].optimizer.step()
        networks['learn_step_counter'] += 1

        self.decrement_epsilon()

        return loss.item()
    

    def learn_DDQN(self, networks):
        networks['q_eval'].optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory(networks)

        indices = T.LongTensor(np.arange(self.batch_size).astype(np.int64))

        q_pred = networks['q_eval'](states)[indices, actions.type(T.LongTensor)]

        q_next = networks['q_next'](states_)
        q_eval = networks['q_eval'](states_)

        max_actions = T.argmax(q_eval, dim = 1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = networks['q_eval'].loss(q_target, q_pred).to(networks['q_eval'].device)

        loss.backward()
        _check_nan(loss, f"DDQN loss at step {networks['learn_step_counter']}")
        
        networks['q_eval'].optimizer.step()

        networks['learn_step_counter']+=1

        self.decrement_epsilon()

        return loss.item()

    def learn_DDPG(self, networks, gsp = False, recurrent = False):
        states, actions, rewards, states_, dones = self.sample_memory(networks)
        target_actions = networks['target_actor'](states_)
        q_value_ = networks['target_critic'](states_, target_actions)

        target = T.unsqueeze(rewards, 1) + self.gamma*q_value_

        #Critic Update
        networks['critic'].optimizer.zero_grad()

        q_value = networks['critic'](states, actions)
        value_loss = Loss(q_value, target)
        value_loss.backward()
        _check_nan(value_loss, f"DDPG critic loss at step {networks['learn_step_counter']}")
        networks['critic'].optimizer.step()

        #Actor Update
        networks['actor'].optimizer.zero_grad()

        new_policy_actions = networks['actor'](states)
        actor_loss = -networks['critic'](states, new_policy_actions)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        _check_nan(actor_loss, f"DDPG actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        return actor_loss.item()
    
    def learn_RDDPG(self, networks, gsp = False, recurrent = False):
        mem_result = self.sample_memory(networks)
        if len(mem_result) == 7:
            states, actions, rewards, states_, dones, h_batch, c_batch = mem_result
            device = networks['actor'].device
            h_0 = T.tensor(np.array(h_batch), dtype=T.float32).to(device)
            c_0 = T.tensor(np.array(c_batch), dtype=T.float32).to(device)
            # h_batch shape: (batch, num_layers, 1, hidden) -> (num_layers, batch, hidden)
            h_0 = h_0.squeeze(2).permute(1, 0, 2).contiguous()
            c_0 = c_0.squeeze(2).permute(1, 0, 2).contiguous()
            hidden_init = (h_0, c_0)
        else:
            states, actions, rewards, states_, dones = mem_result
            hidden_init = None

        # states: (batch, seq_len, obs_dim)
        # actions: (batch, seq_len, act_dim)
        seq_len = states.shape[1]
        burn_in_len = seq_len // 2
        train_len = seq_len - burn_in_len

        # Split into burn-in and training portions
        burn_states = states[:, :burn_in_len, :]
        train_states = states[:, burn_in_len:, :]
        burn_states_ = states_[:, :burn_in_len, :]
        train_states_ = states_[:, burn_in_len:, :]
        train_actions = actions[:, burn_in_len:, :]
        train_rewards = rewards[:, burn_in_len:]

        # Burn-in: refresh hidden state without gradients
        with T.no_grad():
            if burn_in_len > 0:
                # Run burn-in through actor encoder to get hidden state
                _, actor_hidden = networks['actor'].ee(burn_states, hidden=hidden_init)
                _, critic_hidden = networks['critic'].ee(burn_states, hidden=hidden_init)
                _, target_actor_hidden = networks['target_actor'].ee(burn_states_, hidden=hidden_init)
                _, target_critic_hidden = networks['target_critic'].ee(burn_states_, hidden=hidden_init)
            else:
                actor_hidden = hidden_init
                critic_hidden = hidden_init
                target_actor_hidden = hidden_init
                target_critic_hidden = hidden_init

        # Detach hidden states so burn-in gradients don't flow
        if actor_hidden is not None:
            actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
            critic_hidden = (critic_hidden[0].detach(), critic_hidden[1].detach())
            target_actor_hidden = (target_actor_hidden[0].detach(), target_actor_hidden[1].detach())
            target_critic_hidden = (target_critic_hidden[0].detach(), target_critic_hidden[1].detach())

        # Target computation (no gradients)
        with T.no_grad():
            target_actions, _ = networks['target_actor'](train_states_, hidden=target_actor_hidden)
            q_value_, _ = networks['target_critic'](train_states_, target_actions, hidden=target_critic_hidden)
            # Use last timestep for Bellman target
            q_last_ = q_value_[:, -1, :]  # (batch, 1)
            r_last = train_rewards[:, -1]  # (batch,)
            target = r_last.unsqueeze(1) + self.gamma * q_last_  # (batch, 1)

        # Critic update
        networks['critic'].optimizer.zero_grad()
        q_value, _ = networks['critic'](train_states, train_actions, hidden=critic_hidden)
        q_last = q_value[:, -1, :]  # (batch, 1)
        value_loss = Loss(q_last, target)
        value_loss.backward()
        _check_nan(value_loss, f"RDDPG critic loss at step {networks['learn_step_counter']}")
        networks['critic'].optimizer.step()

        # Actor update
        networks['actor'].optimizer.zero_grad()
        new_policy_actions, _ = networks['actor'](train_states, hidden=actor_hidden)
        # Re-run critic with fresh hidden (detached) for actor loss
        actor_q_val, _ = networks['critic'](train_states, new_policy_actions, hidden=critic_hidden)
        actor_loss = -actor_q_val[:, -1, :].mean()
        actor_loss.backward()
        _check_nan(actor_loss, f"RDDPG actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        return actor_loss.item()

    def learn_TD3(self, networks, gsp = False, recurrent = False):
        states, actions, rewards, states_, dones = self.sample_memory(networks)

        with T.no_grad():
            target_actions = networks['target_actor'].forward(states_)
            noise = T.clamp(
                T.tensor(np.random.normal(0, 0.2, size=target_actions.shape).astype(np.float32)),
                -0.5, 0.5
            ).to(target_actions.device)
            target_actions = T.clamp(target_actions + noise, -self.min_max_action, self.min_max_action)

            q1_ = networks['target_critic_1'].forward(states_, target_actions)
            q2_ = networks['target_critic_2'].forward(states_, target_actions)

            q1_[dones] = 0.0
            q2_[dones] = 0.0

            q1_ = q1_.view(-1)
            q2_ = q2_.view(-1)

            critic_value_ = T.min(q1_, q2_)
            target = rewards + self.gamma * critic_value_

        q1 = networks['critic_1'].forward(states, actions).squeeze()
        q2 = networks['critic_2'].forward(states, actions).squeeze()

        networks['critic_1'].optimizer.zero_grad()
        networks['critic_2'].optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss

        critic_loss.backward()
        _check_nan(critic_loss, f"TD3 critic loss at step {networks['learn_step_counter']}")
        networks['critic_1'].optimizer.step()
        networks['critic_2'].optimizer.step()

        networks['learn_step_counter'] += 1

        if networks['learn_step_counter'] % self.update_actor_iter != 0:
            return 0, 0

        networks['actor'].optimizer.zero_grad()
        actor_q1_loss = networks['critic_1'].forward(states, networks['actor'].forward(states))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        _check_nan(actor_loss, f"TD3 actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        self.update_TD3_network_parameters(self.tau, networks)

        return actor_loss.item()

    def learn_attention(self, networks):
        if networks['replay'].mem_ctr < self.gsp_batch_size:
            return 0
        observations, labels = self.sample_attention_memory(networks)
        networks['learn_step_counter'] += 1
        networks['attention'].optimizer.zero_grad()
        pred_headings = networks['attention'](observations)
        loss = Loss(pred_headings, labels.unsqueeze(-1))
        loss.backward()
        _check_nan(loss, f"Attention loss at step {networks['learn_step_counter']}")
        networks['attention'].optimizer.step()
        return loss.item()

    def learn_gsp_mse(self, networks, recurrent: bool = False):
        """Train the GSP prediction network via direct supervised MSE.

        Replaces the DDPG/RDDPG actor-critic path for non-attention GSP variants.
        Samples (state, label) pairs from `networks['replay']`, forwards the state
        through the actor network, and minimizes MSE against the label. The label
        is stored in the action field of the replay buffer by convention — see
        RL-CollectiveTransport Main.py's store_gsp_transition call sites.

        Rationale: see docs/research/2026-04-13-gsp-information-collapse-analysis.md
        in the Stelaris repo. Training the GSP predictor as a DDPG actor-critic
        on a clipped negative-MSE reward produced an information-collapsed
        predictor whose output was worse than predicting the constant mean.
        Direct supervised MSE has a non-vanishing gradient `2(pred-label)` that
        drives the predictor toward the label regardless of how flat the reward
        landscape is.
        """
        if networks['replay'].mem_ctr < self.gsp_batch_size:
            return None

        if recurrent:
            mem_result = self.sample_memory(networks)
            if len(mem_result) == 7:
                states, labels, _, _, _, _, _ = mem_result
            else:
                states, labels, _, _, _ = mem_result
            networks['actor'].optimizer.zero_grad()
            preds_out = networks['actor'](states, hidden=None)
            preds = preds_out[0] if isinstance(preds_out, tuple) else preds_out
            if preds.dim() == labels.dim() + 1:
                labels_shaped = labels.unsqueeze(-1)
            else:
                labels_shaped = labels.view_as(preds)
            loss = F.mse_loss(preds, labels_shaped)
        else:
            states, labels, _, _, _ = self.sample_memory(networks)
            networks['actor'].optimizer.zero_grad()
            preds = networks['actor'].forward(states)
            # labels shape: (batch,) or (batch, 1). preds shape: (batch, 1).
            if labels.dim() == preds.dim() - 1:
                labels_shaped = labels.unsqueeze(-1)
            else:
                labels_shaped = labels.view_as(preds)
            loss = F.mse_loss(preds, labels_shaped)

        loss.backward()
        _check_nan(loss, f"GSP MSE loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()
        networks['learn_step_counter'] += 1
        return loss.item()

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon-self.eps_dec, self.eps_min)

    def store_transition(self, s, a, r, s_, d, networks):
        networks['replay'].store_transition(s, a, r, s_, d)
    
    def store_attention_transition(self, s, y, networks):
        networks['replay'].store_transition(s, y)

    def sample_memory(self, networks):
        result = networks['replay'].sample_buffer(self.batch_size)
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            device = networks['q_eval'].device
        elif networks['learning_scheme'] in {'DDPG', 'RDDPG', 'TD3'}:
            device = networks['actor'].device

        if len(result) == 7:
            states, actions, rewards, states_, dones, h_batch, c_batch = result
            states = T.tensor(states, dtype=T.float32).to(device)
            actions = T.tensor(actions, dtype=T.float32).to(device)
            rewards = T.tensor(rewards, dtype=T.float32).to(device)
            states_ = T.tensor(states_, dtype=T.float32).to(device)
            dones = T.tensor(dones).to(device)
            return states, actions, rewards, states_, dones, h_batch, c_batch
        else:
            states, actions, rewards, states_, dones = result
            states = T.tensor(states, dtype=T.float32).to(device)
            actions = T.tensor(actions, dtype=T.float32).to(device)
            rewards = T.tensor(rewards, dtype=T.float32).to(device)
            states_ = T.tensor(states_, dtype=T.float32).to(device)
            dones = T.tensor(dones).to(device)
            return states, actions, rewards, states_, dones

    def sample_attention_memory(self, networks):
        observations, labels = networks['replay'].sample_buffer(self.batch_size)
        observations = T.tensor(observations, dtype = T.float32).to(networks['attention'].device)
        labels = T.tensor(labels, dtype = T.float32).to(networks['attention'].device)
        return observations, labels