"""Tests for end-to-end Actor.learn() dispatch across all algorithms.

Covers:
- TestEndToEndLearnPipeline: For each algorithm (DQN, DDQN, DDPG, TD3),
  construct Actor, fill buffer with transitions, call learn() — verify no crash.
- TestEpsilonDecays: Verify epsilon decreases after learning steps but
  does not fall below EPS_MIN.
"""

import numpy as np
import pytest

from gsp_rl.src.actors.actor import Actor


BASE_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": 0.001,
    "BETA": 0.002,
    "LR": 0.001,
    "EPSILON": 0.0,
    "EPS_MIN": 0.0,
    "EPS_DEC": 0.0,
    "BATCH_SIZE": 8,
    "MEM_SIZE": 100,
    "REPLACE_TARGET_COUNTER": 10,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 1,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": 8,
}

INPUT_SIZE = 8
OUTPUT_SIZE = 4
N_TRANSITIONS = 30


def make_actor(network, config=None):
    if config is None:
        config = BASE_CONFIG
    return Actor(
        id=1,
        config=config,
        network=network,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        min_max_action=1,
        meta_param_size=1,
        gsp=False,
    )


def fill_buffer(actor, n=N_TRANSITIONS):
    """Store n random transitions into actor's replay buffer."""
    for _ in range(n):
        s = np.random.random(INPUT_SIZE).astype(np.float32)
        a = actor.choose_action(s, actor.networks, test=True)
        r = float(np.random.random())
        s_ = np.random.random(INPUT_SIZE).astype(np.float32)
        d = False
        actor.store_transition(s, a, r, s_, d, actor.networks)


class TestEndToEndLearnPipeline:
    """actor.learn() dispatches correctly for every supported algorithm."""

    def test_dqn_learn_no_crash(self):
        """DQN actor.learn() runs without error after buffer is filled."""
        actor = make_actor("DQN")
        fill_buffer(actor)
        actor.learn()

    def test_ddqn_learn_no_crash(self):
        """DDQN actor.learn() runs without error after buffer is filled."""
        actor = make_actor("DDQN")
        fill_buffer(actor)
        actor.learn()

    def test_ddpg_learn_no_crash(self):
        """DDPG actor.learn() runs without error after buffer is filled."""
        actor = make_actor("DDPG")
        fill_buffer(actor)
        actor.learn()

    def test_td3_learn_no_crash(self):
        """TD3 actor.learn() runs without error after buffer is filled."""
        actor = make_actor("TD3")
        fill_buffer(actor)
        actor.learn()


class TestEpsilonDecays:
    """Epsilon decrements after learn() calls but never below EPS_MIN."""

    def test_epsilon_decays_after_learning(self):
        """Epsilon should decrease after filling buffer and calling learn() multiple times."""
        config = {**BASE_CONFIG, "EPSILON": 1.0, "EPS_MIN": 0.01, "EPS_DEC": 0.01}
        actor = make_actor("DQN", config=config)

        epsilon_before = actor.epsilon
        fill_buffer(actor)

        for _ in range(10):
            actor.learn()

        assert actor.epsilon < epsilon_before

    def test_epsilon_does_not_go_below_eps_min(self):
        """Epsilon should never fall below EPS_MIN regardless of how many steps are run."""
        config = {**BASE_CONFIG, "EPSILON": 1.0, "EPS_MIN": 0.5, "EPS_DEC": 0.1}
        actor = make_actor("DQN", config=config)

        fill_buffer(actor)
        # Run enough steps that epsilon would go below EPS_MIN without clamping
        for _ in range(100):
            actor.learn()

        assert actor.epsilon >= config["EPS_MIN"]
