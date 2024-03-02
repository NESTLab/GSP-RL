import numpy as np
import math
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt

from gsp_rl.src.actors import Actor

'''
Intention will try to guess the angle of the pole at
the next time step based on the carts current position
'''


if __name__ == "__main__":
    ## Discrete Action Spaces
    env = gym.make('CartPole-v1')

    # Network Architecutre
    # network = "DQN"
    network = "DDQN"
    intention_network = "DDPG"
    
    # Build the actor with the following arguments
    nn_args = {
            'id':1,
            'input_size':4,
            'output_size':2,
            'meta_param_size':2, 
            'intention':True,
            'recurrent_intention':False,
            'attention': True,
            'intention_input_size': 1,
            'intention_output_size': 1,
            'intention_min_max_action': 1.0,
            'intention_look_back':2,
            'intention_sequence_length': 5
    }

    actor = Actor(**nn_args)
    actor.build_networks(network)
    actor.build_intention_network(intention_network)
    scores, intention_scores, avg_exp_scores, eps_history = [], [], [], []
    avg_exp_intention_scores = []
    n_games = 1000

    for i in range(n_games):
        time = 0
        score = 0
        intention_reward = 0
        done = False
        observation, _ = env.reset()
        intention_observation = observation[0] 
        next_predicted_pole_angle_change = actor.choose_action([intention_observation], actor.intention_networks)
        if intention_network is not None:
            observation = np.append(observation, next_predicted_pole_angle_change)
        while not done:
            action = actor.choose_action(observation, actor.networks)
            # action = env.action_space.sample()
            observation_, reward, done, truncated, info = env.step(action)

            # Calculate angle difference between time steps for intention learning
            if intention_network is not None:
                old_pole_angle = intention_observation + math.pi
                new_pole_angle = observation_[2] + math.pi
                if old_pole_angle < math.pi and new_pole_angle > math.pi:
                    diff = old_pole_angle - new_pole_angle +  2*math.pi
                elif old_pole_angle < math.pi and new_pole_angle < math.pi:
                    diff = new_pole_angle - old_pole_angle + 2*math.pi
                else:
                    diff = old_pole_angle - new_pole_angle
                
                label = diff
                x1 = math.cos(diff)
                y1 = math.sin(diff)
                x2 = math.cos(next_predicted_pole_angle_change)
                y2 = math.sin(next_predicted_pole_angle_change)

                dot = np.dot([x1, y1], [x2, y2])
                intention_step_reward = dot
            
            score += reward
            intention_reward += intention_step_reward
            done = done or truncated
            if nn_args['attention']:
                actor.store_intention_transition(intention_observation, label, 0, 0, 0)
            else:
                actor.store_intention_transition(
                        intention_observation,
                        next_predicted_pole_angle_change,
                        intention_step_reward,
                        observation_[0],
                        done
                )
            
            intention_observation = observation_[0]
            next_predicted_pole_angle_change = actor.choose_action([intention_observation], actor.intention_networks)
            if intention_network is not None:
                observation_ = np.append(observation_, next_predicted_pole_angle_change)
            actor.store_agent_transition(observation, [action], reward, observation_, done)
            actor.learn()
            observation = observation_
            time+=1
        
        scores.append(score)
        intention_scores.append(intention_reward/time)
        eps_history.append(actor.epsilon)
        
        avg_score = np.mean(scores[-10:])
        avg_intention_score = np.mean(intention_scores[-10:])
        if i%10 == 0:
            avg_exp_scores.append(avg_score)
            avg_exp_intention_scores.append(avg_intention_score)
        
        print(f"Episode: {i}, Epsilon: {actor.epsilon}, Score {score}, Average Score: {avg_score}, Intention Score: {intention_reward/score: .2f}, Average Intention Score: {avg_intention_score: .2f}")
        
    x = [i+1 for i in range(n_games)]
    x_avg = [(i+1)*10 for i in range(math.floor(n_games/10))]

    plt.scatter(x, scores)
    plt.plot(x_avg, avg_exp_scores, c='r')
    plt.title(f"Cart Pole GSP-RL {network} Agent Baseline")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.savefig(f'Cart_Pole_{network}_baseline.png')

    plt.clf()
    plt.scatter(x, intention_scores)
    plt.plot(x_avg, avg_exp_intention_scores, c='r')
    plt.title(f"Cart Pole GSP-RL {intention_network} Intention Agent Baseline")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.savefig(f'Cart_Pole_{intention_network}_intention_mixed_reward.png')
        