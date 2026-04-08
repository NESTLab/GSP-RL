"""Tests for batched action selection — verify it produces same results as sequential."""

import numpy as np
import torch as T
import pytest

from gsp_rl.src.actors.actor import Actor


@pytest.fixture
def config():
    return {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }


class TestDQNBatch:
    def test_batch_matches_sequential(self, config):
        """Batched DQN should return identical actions to sequential calls."""
        config["EPSILON"] = 0.0  # greedy — no randomness
        actor = Actor(id=1, config=config, network="DQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]

        sequential = [actor.choose_action(obs, actor.networks, test=True) for obs in observations]
        batched = actor.choose_actions_batch(observations, actor.networks, test=True)

        assert sequential == batched

    def test_batch_returns_correct_count(self, config):
        actor = Actor(id=1, config=config, network="DQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(6)]
        actions = actor.choose_actions_batch(observations, actor.networks, test=True)
        assert len(actions) == 6

    def test_batch_actions_in_range(self, config):
        actor = Actor(id=1, config=config, network="DQN",
                      input_size=8, output_size=9, min_max_action=1, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]
        actions = actor.choose_actions_batch(observations, actor.networks, test=True)
        for a in actions:
            assert 0 <= a < 9


class TestDDQNBatch:
    def test_batch_matches_sequential(self, config):
        config["EPSILON"] = 0.0
        actor = Actor(id=1, config=config, network="DDQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]

        sequential = [actor.choose_action(obs, actor.networks, test=True) for obs in observations]
        batched = actor.choose_actions_batch(observations, actor.networks, test=True)

        assert sequential == batched


class TestDDPGBatch:
    def test_batch_matches_sequential(self, config):
        config["NOISE"] = 0.0
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]

        sequential = np.array([actor.choose_action(obs, actor.networks, test=True) for obs in observations])
        batched = actor.choose_actions_batch(observations, actor.networks, test=True)

        np.testing.assert_allclose(sequential, batched, atol=1e-5)

    def test_batch_output_shape(self, config):
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]
        actions = actor.choose_actions_batch(observations, actor.networks, test=True)
        assert actions.shape == (4, 2)


class TestTD3Batch:
    def test_batch_matches_sequential_after_warmup(self, config):
        config["NOISE"] = 0.0
        config["WARMUP"] = 0
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        # Advance past warmup
        actor.time_step = 100
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]

        # Sequential calls each increment time_step, so save/restore
        saved_ts = actor.time_step
        sequential = []
        for obs in observations:
            actor.time_step = saved_ts  # reset to same point
            sequential.append(actor.choose_action(obs, actor.networks, test=True))
        sequential = np.array(sequential)

        actor.time_step = saved_ts
        batched = actor.choose_actions_batch(observations, actor.networks, test=True)

        np.testing.assert_allclose(sequential, batched, atol=1e-5)

    def test_batch_output_shape(self, config):
        config["WARMUP"] = 0
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        actor.time_step = 100
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]
        actions = actor.choose_actions_batch(observations, actor.networks, test=True)
        assert actions.shape == (4, 2)


class TestBatchNotSupported:
    def test_rddpg_raises(self, config):
        """RDDPG should explicitly refuse batching."""
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1,
                      gsp=True, recurrent_gsp=True, gsp_input_size=6, gsp_output_size=1)
        observations = [np.random.randn(8).astype(np.float32) for _ in range(4)]
        # The main networks are DDPG (batchable), but gsp_networks are RDDPG
        with pytest.raises(NotImplementedError):
            actor.choose_actions_batch(observations, actor.gsp_networks, test=True)
