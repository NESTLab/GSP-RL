import numpy as np
import math
import os
import yaml
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt

from gsp_rl.src.actors import Actor

class CartPoleAgent(Actor):
    def __init__(
            self,
            nn_args,
    ):
        self.nn_args = nn_args
        super().__init__(**nn_args)
        if nn_args['recurrent_gsp']:
            self.gsp_observation = [[0 for _ in range(nn_args['gsp_input_size'])] for _ in range(nn_args['gsp_sequence_length'])]
    
    def make_agent_state(self, state, gsp_state = None):
        if gsp_state is not None:
            return np.append(state, gsp_state)
        return state

    def make_gsp_state(self, gsp_state):
        if not self.nn_args['gsp']:
            return None
        if self.nn_args['recurrent_gsp']:
            self.gsp_observation.append(gsp_state)
            self.gsp_observation.pop(0)
            return self.gsp_observation
        return gsp_state
    

    def choose_gsp_action(self, gsp_state):
        action = self.choose_action(gsp_state, self.gsp_networks)
        if self.nn_args['recurrent_gsp']:
            return action[-1]
        return action
    
    def build_gsp_reward(self, pred, label):
        x1 = math.cos(label)
        y1 = math.sin(label)
        x2 = math.cos(pred.item())
        y2 = math.sin(pred.item())

        return np.dot([x1, y1], [x2, y2])

'''
gsp will try to guess the angle of the pole at
the next time step based on the carts current position
'''


if __name__ == "__main__":
    ## Discrete Action Spaces
    env = gym.make('CartPole-v1')
    
    # Build the actor with the following arguments
    containing_folder = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(containing_folder, 'cart_pole_config.yml')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    nn_args = {
            'network': config['LEARNING_SCHEME'],
            'id':1,
            'input_size':config['INPUT_SIZE'],
            'output_size':config['OUTPUT_SIZE'],
            'min_max_action':config['MIN_MAX_ACTION'],
            'meta_param_size':config['META_PARAM_SIZE'], 
            'gsp':config['GSP'],
            'recurrent_gsp':config['RECURRENT'],
            'attention': config['ATTENTION'],
            'gsp_input_size': config['GSP_INPUT_SIZE'],
            'gsp_output_size': config['GSP_OUTPUT_SIZE'],
            'gsp_min_max_action': config['GSP_MIN_MAX_ACTION'],
            'gsp_look_back':config['GSP_LOOK_BACK'],
            'gsp_sequence_length': config['GSP_SEQUENCE_LENGTH'],
            'config': config
    }

    agent = CartPoleAgent(nn_args)
    scores, gsp_scores, avg_exp_scores, eps_history = [], [], [], []
    avg_exp_gsp_scores = []
    n_games = config['N_GAMES']

    for i in range(n_games):
        time = 0
        score = 0
        gsp_reward = 0
        done = False
        observation, _ = env.reset()
        next_predicted_pole_angle_change = None
        if nn_args['gsp']:
            gsp_observation = agent.make_gsp_state([observation[0]])
            next_predicted_pole_angle_change = agent.choose_action(gsp_observation, agent.gsp_networks)[-1]
        observation = agent.make_agent_state(observation, next_predicted_pole_angle_change)
        while not done:
            action = agent.choose_action(observation, agent.networks)
            # action = env.action_space.sample()
            observation_, reward, done, truncated, info = env.step(action)

            # Calculate angle difference between time steps for gsp learning
            if nn_args['gsp']:
                old_pole_angle = observation[0] + math.pi
                new_pole_angle = observation_[2] + math.pi
                if old_pole_angle < math.pi and new_pole_angle > math.pi:
                    diff = old_pole_angle - new_pole_angle +  2*math.pi
                elif old_pole_angle < math.pi and new_pole_angle < math.pi:
                    diff = new_pole_angle - old_pole_angle + 2*math.pi
                else:
                    diff = old_pole_angle - new_pole_angle
                
                label = diff
                
                gsp_step_reward = agent.build_gsp_reward(next_predicted_pole_angle_change, label)
                gsp_reward += gsp_step_reward
            
            score += reward
            done = done or truncated
            if nn_args['gsp']:
                if nn_args['attention']:
                    agent.store_gsp_transition(gsp_observation, label, 0, 0, 0)
                else:
                    agent.store_gsp_transition(
                            observation[0],
                            next_predicted_pole_angle_change,
                            gsp_step_reward,
                            observation_[0],
                            done
                    )
                gsp_observation = agent.make_gsp_state([observation_[0]])
                next_predicted_pole_angle_change = agent.choose_action(gsp_observation, agent.gsp_networks)[-1]

            observation_ = agent.make_agent_state(observation_, next_predicted_pole_angle_change)
            agent.store_agent_transition(observation, [action], reward, observation_, done)
            agent.learn()
            observation = observation_
            time+=1
        
        scores.append(score)
        gsp_scores.append(gsp_reward/time)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-10:])
        avg_gsp_score = np.mean(gsp_scores[-10:])
        if i%10 == 0:
            avg_exp_scores.append(avg_score)
            avg_exp_gsp_scores.append(avg_gsp_score)
        
        print(f"Episode: {i}, Epsilon: {agent.epsilon}, Score {score}, Average Score: {avg_score}, gsp Score: {gsp_reward/score: .2f}, Average gsp Score: {avg_gsp_score: .2f}")
        
    x = [i+1 for i in range(n_games)]
    x_avg = [(i+1)*10 for i in range(math.floor(n_games/10))]

    if nn_args['gsp']:
        gsp_network = 'DDPG'
        if nn_args['recurrent_gsp']:
            gsp_network = "RDDPG"
        elif nn_args['attention']:
            gsp_network = "Attention"

    plt.scatter(x, scores)
    plt.plot(x_avg, avg_exp_scores, c='r')
    plt.title(f"Cart Pole GSP-RL {network} Agent Baseline")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    if nn_args['gsp']:
        plt.savefig(f'plots/Cart_Pole_{network}_baseline_reward_with_{gsp_network}_GSP.png')
    else:
        plt.savefig(f'plots/Cart_Pole_{network}_baseline.png')
    if nn_args['gsp']:
        plt.clf()
        plt.scatter(x, gsp_scores)
        plt.plot(x_avg, avg_exp_gsp_scores, c='r')
        plt.title(f"Cart Pole GSP-RL {gsp_network} gsp Agent Baseline")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.savefig(f'plots/Cart_Pole_{network}_baseline_with_{gsp_network}_GSP_reward.png')
        