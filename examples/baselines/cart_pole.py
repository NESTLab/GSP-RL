import numpy as np
import math
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
        if nn_args['recurrent_intention']:
            self.intention_observation = [[0 for _ in range(nn_args['intention_input_size'])] for _ in range(nn_args['intention_sequence_length'])]
    
    def make_agent_state(self, state, intention_state = None):
        if intention_state is not None:
            return np.append(state, intention_state)
        return state

    def make_gsp_state(self, gsp_state):
        if not self.nn_args['intention']:
            return None
        if self.nn_args['recurrent_intention']:
            self.intention_observation.append(gsp_state)
            self.intention_observation.pop(0)
            return self.intention_observation
        return gsp_state
    

    def choose_gsp_action(self, gsp_state):
        action = self.choose_action(gsp_state, self.intention_networks)
        if self.nn_args['recurrent_intention']:
            return action[-1]
        return action
    
    def build_gsp_reward(self, pred, label):
        x1 = math.cos(label)
        y1 = math.sin(label)
        x2 = math.cos(pred.item())
        y2 = math.sin(pred.item())

        return np.dot([x1, y1], [x2, y2])

'''
Intention will try to guess the angle of the pole at
the next time step based on the carts current position
'''


if __name__ == "__main__":
    ## Discrete Action Spaces
    env = gym.make('CartPole-v1')

    # Network Learning Scheme
    network = "DDQN"
    
    # Build the actor with the following arguments
    nn_args = {
            'network': network,
            'id':1,
            'input_size':4,
            'output_size':2,
            'meta_param_size':256, 
            'intention':True,
            'recurrent_intention':False,
            'attention': True,
            'intention_input_size': 1,
            'intention_output_size': 1,
            'intention_min_max_action': 1.0,
            'intention_look_back':2,
            'intention_sequence_length': 5
    }

    agent = CartPoleAgent(nn_args)
    scores, intention_scores, avg_exp_scores, eps_history = [], [], [], []
    avg_exp_intention_scores = []
    n_games = 2000

    for i in range(n_games):
        time = 0
        score = 0
        intention_reward = 0
        done = False
        observation, _ = env.reset()
        next_predicted_pole_angle_change = None
        if nn_args['intention']:
            intention_observation = agent.make_gsp_state([observation[0]])
            next_predicted_pole_angle_change = agent.choose_action(intention_observation, agent.intention_networks)[-1]
        observation = agent.make_agent_state(observation, next_predicted_pole_angle_change)
        while not done:
            action = agent.choose_action(observation, agent.networks)
            # action = env.action_space.sample()
            observation_, reward, done, truncated, info = env.step(action)

            # Calculate angle difference between time steps for intention learning
            if nn_args['intention']:
                old_pole_angle = observation[0] + math.pi
                new_pole_angle = observation_[2] + math.pi
                if old_pole_angle < math.pi and new_pole_angle > math.pi:
                    diff = old_pole_angle - new_pole_angle +  2*math.pi
                elif old_pole_angle < math.pi and new_pole_angle < math.pi:
                    diff = new_pole_angle - old_pole_angle + 2*math.pi
                else:
                    diff = old_pole_angle - new_pole_angle
                
                label = diff
                
                intention_step_reward = agent.build_gsp_reward(next_predicted_pole_angle_change, label)
                intention_reward += intention_step_reward
            
            score += reward
            done = done or truncated
            if nn_args['intention']:
                if nn_args['attention']:
                    agent.store_intention_transition(intention_observation, label, 0, 0, 0)
                else:
                    agent.store_intention_transition(
                            observation[0],
                            next_predicted_pole_angle_change,
                            intention_step_reward,
                            observation_[0],
                            done
                    )
                intention_observation = agent.make_gsp_state([observation_[0]])
                next_predicted_pole_angle_change = agent.choose_action(intention_observation, agent.intention_networks)[-1]

            observation_ = agent.make_agent_state(observation_, next_predicted_pole_angle_change)
            agent.store_agent_transition(observation, [action], reward, observation_, done)
            agent.learn()
            observation = observation_
            time+=1
        
        scores.append(score)
        intention_scores.append(intention_reward/time)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-10:])
        avg_intention_score = np.mean(intention_scores[-10:])
        if i%10 == 0:
            avg_exp_scores.append(avg_score)
            avg_exp_intention_scores.append(avg_intention_score)
        
        print(f"Episode: {i}, Epsilon: {agent.epsilon}, Score {score}, Average Score: {avg_score}, Intention Score: {intention_reward/score: .2f}, Average Intention Score: {avg_intention_score: .2f}")
        
    x = [i+1 for i in range(n_games)]
    x_avg = [(i+1)*10 for i in range(math.floor(n_games/10))]

    if nn_args['intention']:
        intention_network = 'DDPG'
        if nn_args['recurrent_intention']:
            intention_network = "RDDPG"
        elif nn_args['attention']:
            intention_network = "Attention"

    plt.scatter(x, scores)
    plt.plot(x_avg, avg_exp_scores, c='r')
    plt.title(f"Cart Pole GSP-RL {network} Agent Baseline")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    if nn_args['intention']:
        plt.savefig(f'plots/Cart_Pole_{network}_baseline_reward_with_{intention_network}_GSP.png')
    else:
        plt.savefig(f'plots/Cart_Pole_{network}_baseline.png')
    if nn_args['intention']:
        plt.clf()
        plt.scatter(x, intention_scores)
        plt.plot(x_avg, avg_exp_intention_scores, c='r')
        plt.title(f"Cart Pole GSP-RL {intention_network} Intention Agent Baseline")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.savefig(f'plots/Cart_Pole_{network}_baseline_with_{intention_network}_GSP_reward.png')
        