from gsp_rl.src.networks import (
    DQN,
    DDQN,
    DDPGActorNetwork,
    DDPGCriticNetwork,
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

Loss = nn.MSELoss()


class Hyperparameters:
    def __init__(self):
        self.gamma = 0.99997
        self.tau = 0.005
        self.alpha = 0.001
        self.beta = 0.002
        self.lr = 0.0001
        self.ee_lr = 0.01
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-5

        self.intn_epsilon = 1.0
        self.intn_eps_min = 0.01
        self.intn_eps_dec = 1e-5
        self.intn_learning_offset = 1000 #learn after every 1000 action network learning steps

        self.batch_size = 100
        self.mem_size = 100000
        self.replace_target_ctr = 1000
        self.failed = False
        self.failure_action = [0, 0, 1]

        self.noise = 0.1
        self.update_actor_iter = 2
        self.warmup = 1000
        self.time_step = 0

        self.min_max_action = 1

        #GNN
        self.hidden_channels = 32
        self.num_heads = 4

class NetworkAids(Hyperparameters):
    def __init__(self):
        super().__init__()
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
    
    def make_EE_networks(self, nn_args):
        return EnvironmentEncoder(**nn_args)

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
    
    def DDPG_choose_action(self, observation, networks):
        state = T.tensor(observation, dtype = T.float).to(networks['actor'].device)
        if networks['learning_scheme'] == 'RDDPG':
            s,s_,a,r,d = networks['replay'].get_current_sequence()
            s,a,s_,r = self.build_initial_ee_input(s,a,s_,r)
            obs_padded = self.build_ee_input(s,a,s_,r).to(networks['ee'].device)
            mp = networks['ee'](obs_padded, True)
            state = T.cat((state, mp))
            return networks['actor'].forward(state).unsqueeze(0)
        return networks['actor'].forward(state).unsqueeze(0)
        
    
    def TD3_choose_action(self, observation, networks, n_actions):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale = self.noise,
                                           size = (n_actions,))
                          ).to(networks['actor'].device)
        else:
            state = T.tensor(observation, dtype = T.float).to(networks['actor'].device)
            mu = networks['actor'].forward(state).to(networks['actor'].device)
        mu_prime = mu + T.tensor(np.random.normal(scale = self.noise), dtype = T.float).to(networks['actor'].device)
        mu_prime = T.clamp(mu_prime, -self.min_max_action, self.min_max_action)
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()
    
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
        
        networks['q_eval'].optimizer.step()

        networks['learn_step_counter']+=1

        self.decrement_epsilon()

        return loss.item()

    def learn_DDPG(self, networks, intention = False, recurrent = False):
        states, actions, rewards, states_, dones = self.sample_memory(networks)
        if not intention:
            actions = actions[:,:2]
        elif not recurrent:
            actions = actions.unsqueeze(1)

        if intention and recurrent:
            meta_param_obs = self.build_ee_input(states, actions, states_, rewards)
            meta_param = networks['ee'](meta_param_obs)
            meta_param_clone = T.clone(meta_param).detach()
            states = self.build_ac_input(states, meta_param)
            states_ = self.build_ac_input(states_, meta_param_clone)
            states_clone = T.clone(states).detach()
            #reformatting inputs
            actions = actions[:,-1,:]
            rewards = rewards[:,-1]

        target_actions = networks['target_actor'](states_)
        q_value_ = networks['target_critic']([states_, target_actions])
        # import ipdb; ipdb.set_trace()
        #TODO what is dones doing ?
        # q_value_[dones] = 0.0
        target = T.unsqueeze(rewards, 1) + self.gamma*q_value_

        #Critic Update
        networks['critic'].zero_grad()
        q_value = networks['critic']([states, actions])
        value_loss = Loss(q_value, target)
        value_loss.backward()
        networks['critic'].optimizer.step()

        #Actor Update
        networks['actor'].zero_grad()
        if intention and recurrent:
            new_policy_actions = networks['actor'](states_clone)
            actor_loss = -networks['critic']([states_clone, new_policy_actions])
        else:
            new_policy_actions = networks['actor'](states)
            actor_loss = -networks['critic']([states, new_policy_actions])
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        return actor_loss.item()

    def learn_TD3(self, networks, intention = False):
        states, actions, rewards, states_, dones = self.sample_memory(networks)

        if not intention:
            actions = actions[:,:2]
        else:
            actions.unsqueeze(1)

        target_actions = networks['target_actor'].forward(states_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale = 0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, -self.min_max_action, self.min_max_action)

        q1_ = networks['target_critic_1'].forward(states_, target_actions)
        q2_ = networks['target_critic_2'].forward(states_, target_actions)

        q1 = networks['critic_1'].forward(states, actions).squeeze() # need to squeeze to change shape from [100,1] to [100] to match target shape
        q2 = networks['critic_2'].forward(states, actions).squeeze()

        q1_[dones] = 0.0
        q2_[dones] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = rewards + self.gamma*critic_value_

        networks['critic_1'].optimizer.zero_grad()
        networks['critic_2'].optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        networks['critic_1'].optimizer.step()
        networks['critic_2'].optimizer.step()

        networks['learn_step_counter'] += 1

        if networks['learn_step_counter'] % self.update_actor_iter != 0:
            return 0, 0
        #print('Actor Learn Step')
        networks['actor'].optimizer.zero_grad()
        actor_q1_loss = networks['critic_1'].forward(states, networks['actor'].forward(states))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        networks['actor'].optimizer.step()

        return actor_loss.item()

    def learn_attention(self, networks):
        if networks['replay'].mem_ctr < self.batch_size:
            return 0
        observations, labels = self.sample_attention_memory(networks)
        networks['learn_step_counter'] += 1
        networks['attention'].zero_grad()
        pred_headings = networks['attention'](observations)
        loss = Loss(pred_headings, labels.unsqueeze(-1))
        loss.backward()
        networks['attention'].optimizer.step()
        return loss.item()


    def build_initial_state_ee_input(self,state):
        #Padding single inputs, EE takes in batches.
        stateP = self.pad_input(state,self.seq_len,self.batch_size)
        return stateP


#TODO here for later use to build (s,a,s',r,done) input
    def build_initial_ee_input(self,s,a,s_,r):
        s = T.from_numpy(s)
        s_ = T.from_numpy(s_)
        a = T.from_numpy(a)
        r = T.from_numpy(r).unsqueeze(-1)
        #Padding single inputs, EE takes in batches.
        stateP = self.pad_input(s,0,self.batch_size)
        actionP = self.pad_input(a,0,self.batch_size)
        state_P = self.pad_input(s_,0,self.batch_size)
        rewardP = F.pad(r.unsqueeze(0), pad=(0,0,0,0,self.batch_size-1,0), value=0)
        return stateP, actionP, state_P, rewardP

    def pad_input(self,s,seqlen,batch):
        s = s.unsqueeze(0)
        s = F.pad(s,pad=(0,0,0,0,batch-1,0))
        return s
        
    def build_ee_input(self, s, a, s_, r):
        observation = T.cat((s, a, s_), -1)
        if r.dim() == 0:
            r.reshape([1])
        if r.dim() == 2:
            r = r.unsqueeze(-1)
        observation = T.cat((observation, r), -1)
        if observation.dim() == 1:
            observation = T.reshape(observation, (1, 1, observation.shape[0]))
        return observation.to(T.float32)
    
    def build_ac_input(self, state, mp):
        state = state[:, -1, :]
        obs = T.cat((state, mp), 1)
        return obs
        
    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon-self.eps_dec, self.eps_min)

    def store_transition(self, s, a, r, s_, d, networks):
        networks['replay'].store_transition(s, a, r, s_, d)
    
    def store_attention_transition(self, s, y, networks):
        networks['replay'].store_transition(s, y)

    def sample_memory(self, networks):
        states, actions, rewards, states_, dones = networks['replay'].sample_buffer(self.batch_size)
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            device = networks['q_eval'].device
        elif networks['learning_scheme'] in {'DDPG', 'RDDPG', 'TD3'}:
            device = networks['actor'].device
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