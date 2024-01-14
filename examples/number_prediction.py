from gsp_rl.src.actors import NetworkAids
from gsp_rl.src.buffers import ReplayBuffer

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make('CartPole-v1')
nn_args = {
            'id':1,
            'lr':1e-4,
            'input_size':3,
            'output_size':10,
            'fc1_dims': 64,
            'fc2_dims':128
        }
NA = NetworkAids()
DQN_networks = NA.make_DQN_networks(nn_args)
DQN_networks['replay'] = ReplayBuffer(10000, nn_args['input_size'], nn_args['output_size'], 'Discrete')
DQN_networks['learn_step_counter'] = 0
DQN_networks['learning_scheme'] = 'DQN'
exp_reward = []
for episode in range(2000):
    if episode % 10 == 0:
        print(episode)
    ep_reward = 0
    obs_init = np.random.randint(0, nn_args['output_size']-3)
    obs = [obs_init, obs_init+1, obs_init+2]
    for t in range(100):
        predicted_next_number = NA.DQN_DDQN_choose_action(obs, DQN_networks)
        reward = -1
        if predicted_next_number == obs[-1]+1:
            reward = 0
        ep_reward += reward
        next_obs_init = np.random.randint(0, nn_args['output_size']-3)
        next_obs = [obs_init, obs_init+1, obs_init+2]
        NA.store_transition(obs, [predicted_next_number, 1], reward, next_obs, False, DQN_networks)
        obs = next_obs
        if DQN_networks['replay'].mem_ctr > 16:
            NA.learn_DQN(DQN_networks)
    exp_reward.append(ep_reward)

plt.plot(exp_reward)
plt.show()
