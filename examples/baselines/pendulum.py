import numpy as np
import math
import os
import yaml

import gymnasium as gym
import matplotlib.pyplot as plt

from gsp_rl.src.actors import Actor

if __name__ == "__main__":

    ## Continuous Action Spaces
    env = gym.make('Pendulum-v1', g=9.81)

    containing_folder = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(containing_folder, 'pendulum_config.yml')

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

    actor = Actor(**nn_args)
    scores, avg_exp_scores, eps_history = [], [], []
    n_games = config['N_GAMES']

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = actor.choose_action(observation, actor.networks)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            done = done or truncated
            actor.store_agent_transition(observation, action, reward, observation_, done)
            actor.learn()
            observation = observation_
        
        scores.append(score)
        eps_history.append(actor.epsilon)
        
        avg_score = np.mean(scores[-10:])
        if i%10 == 0:
            avg_exp_scores.append(avg_score)
        
        print(f"Episode: {i}, Epsilon: {actor.epsilon}, Score {score}, Average Score: {avg_score}")
        
    x = [i+1 for i in range(n_games)]
    x_avg = [(i+1)*10 for i in range(math.floor(n_games/10))]

    plt.scatter(x, scores)
    plt.plot(x_avg, avg_exp_scores, c='r')
    plt.title(f"Pendulum GSP-RL {network} Agent Baseline")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.savefig(f'Pendulum_{network}_baseline.png')
        
        