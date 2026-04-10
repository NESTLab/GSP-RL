import numpy as np
import pytest
import torch
import gymnasium as gym
from gsp_rl.src.actors.actor import Actor

SEED = 42


def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_config():
    return {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.002,
        "LR": 0.001, "EPSILON": 1.0, "EPS_MIN": 0.01, "EPS_DEC": 0.002,
        "BATCH_SIZE": 64, "MEM_SIZE": 10000, "REPLACE_TARGET_COUNTER": 100,
        "NOISE": 0.1, "UPDATE_ACTOR_ITER": 2, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 1000, "GSP_BATCH_SIZE": 16,
    }


def _train_cartpole(scheme, max_episodes=150):
    _seed_all(SEED)
    config = _make_config()
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]  # 4
    n_actions = env.action_space.n  # 2

    actor = Actor(id=1, config=config, network=scheme,
                  input_size=obs_size, output_size=n_actions,
                  min_max_action=1, meta_param_size=1)

    episode_rewards = []
    for ep in range(max_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        total_reward = 0
        done = False
        while not done:
            action = actor.choose_action(obs.astype(np.float32), actor.networks)
            obs_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            actor.store_transition(obs.astype(np.float32), action, reward,
                                   obs_.astype(np.float32), done, actor.networks)
            actor.learn()
            obs = obs_
            total_reward += reward
        episode_rewards.append(total_reward)
    env.close()
    return episode_rewards


@pytest.mark.slow
class TestCartPoleConvergence:
    def test_dqn_solves_cartpole(self, convergence_plots):
        rewards = _train_cartpole("DQN", max_episodes=150)
        convergence_plots["DQN_CartPole"] = rewards
        avg_last_20 = np.mean(rewards[-20:])
        assert avg_last_20 > 200, f"DQN failed: avg last 20 = {avg_last_20:.1f} (need > 200)"

    def test_ddqn_solves_cartpole(self, convergence_plots):
        rewards = _train_cartpole("DDQN", max_episodes=150)
        convergence_plots["DDQN_CartPole"] = rewards
        avg_last_20 = np.mean(rewards[-20:])
        assert avg_last_20 > 200, f"DDQN failed: avg last 20 = {avg_last_20:.1f} (need > 200)"
