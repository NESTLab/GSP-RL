import numpy as np
import math

import gymnasium as gym
import matplotlib.pyplot as plt

from gsp_rl.src.actors import Actor

if __name__ == "__main__":
    ## Discrete Action Spaces
    env = gym.make('CartPole-v1')

    # Network Architecutre
    # network = "DQN"
    network = "DDQN"
    
    # Build the actor with the following arguments
    actor = Actor(
        id=1,
        input_size=4,
        output_size=2,
        meta_param_size=2
    )
    actor.build_networks(network)
    scores, avg_exp_scores, eps_history = [], [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = actor.choose_action(observation, actor.networks)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            done = done or truncated
            actor.store_agent_transition(observation, [action], reward, observation_, done)
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
    plt.title(f"Cart Pole GSP-RL {network} Agent Baseline")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.savefig(f'Cart_Pole_{network}_baseline.png')
        