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
        "GAMMA": 0.98, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 1.0, "EPS_MIN": 0.01, "EPS_DEC": 0.0,
        "BATCH_SIZE": 100, "MEM_SIZE": 50000, "REPLACE_TARGET_COUNTER": 100,
        "NOISE": 0.1, "UPDATE_ACTOR_ITER": 2, "WARMUP": 200,
        "GSP_LEARNING_FREQUENCY": 1000, "GSP_BATCH_SIZE": 16,
    }


def _random_baseline(max_episodes=20):
    env = gym.make("Pendulum-v1")
    rewards = []
    for ep in range(max_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        total = 0
        for _ in range(200):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            if term or trunc:
                break
        rewards.append(total)
    env.close()
    return np.mean(rewards)


def _train_pendulum(scheme, max_episodes=100):
    _seed_all(SEED)
    config = _make_config()
    env = gym.make("Pendulum-v1")
    obs_size = env.observation_space.shape[0]  # 3
    n_actions = env.action_space.shape[0]  # 1
    max_action = float(env.action_space.high[0])  # 2.0

    actor = Actor(id=1, config=config, network=scheme,
                  input_size=obs_size, output_size=n_actions,
                  min_max_action=max_action, meta_param_size=1)

    episode_rewards = []
    for ep in range(max_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 200:
            action = actor.choose_action(obs.astype(np.float32), actor.networks)
            action_env = np.clip(action.flatten(), -max_action, max_action)
            obs_, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated
            actor.store_transition(obs.astype(np.float32), action.flatten(), reward,
                                   obs_.astype(np.float32), done, actor.networks)
            actor.learn()
            obs = obs_
            total_reward += reward
            steps += 1
        episode_rewards.append(total_reward)
    env.close()
    return episode_rewards


@pytest.mark.slow
class TestPendulumConvergence:
    def test_ddpg_improves_over_random(self, convergence_plots):
        random_baseline = _random_baseline()
        rewards = _train_pendulum("DDPG", max_episodes=100)
        convergence_plots["DDPG_Pendulum"] = rewards
        avg_last_20 = np.mean(rewards[-20:])
        improvement = (avg_last_20 - random_baseline) / abs(random_baseline)
        assert improvement > 0.2, (
            f"DDPG failed: avg last 20 = {avg_last_20:.1f}, random = {random_baseline:.1f}, "
            f"improvement = {improvement:.1%}")

    def test_td3_improves_over_random(self, convergence_plots):
        random_baseline = _random_baseline()
        rewards = _train_pendulum("TD3", max_episodes=100)
        convergence_plots["TD3_Pendulum"] = rewards
        avg_last_20 = np.mean(rewards[-20:])
        improvement = (avg_last_20 - random_baseline) / abs(random_baseline)
        assert improvement > 0.2, (
            f"TD3 failed: avg last 20 = {avg_last_20:.1f}, random = {random_baseline:.1f}, "
            f"improvement = {improvement:.1%}")
