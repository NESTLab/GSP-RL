"""Tests for vectorized learn_RDDPG."""

import numpy as np
import torch as T
import time
import pytest

from gsp_rl.src.actors.actor import Actor


@pytest.fixture
def config():
    return {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 1000, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 10, "GSP_BATCH_SIZE": 8,
    }


def _fill_gsp_buffer(actor, n=100):
    for _ in range(n):
        gs = np.random.randn(6).astype(np.float32)
        ga = np.random.randn(1).astype(np.float32)
        actor.store_transition(gs, ga, float(np.random.randn()),
                              np.random.randn(6).astype(np.float32), False,
                              actor.gsp_networks)


class TestVectorizedRDDPG:
    def test_learn_completes(self, config):
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1,
                      gsp=True, recurrent_gsp=True, gsp_input_size=6, gsp_output_size=1,
                      gsp_sequence_length=10, recurrent_hidden_size=32, recurrent_num_layers=2)
        _fill_gsp_buffer(actor, 200)
        loss = actor.learn_RDDPG(actor.gsp_networks, gsp=True, recurrent=True)
        assert np.isfinite(loss)

    def test_weights_change(self, config):
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1,
                      gsp=True, recurrent_gsp=True, gsp_input_size=6, gsp_output_size=1,
                      gsp_sequence_length=10, recurrent_hidden_size=32, recurrent_num_layers=2)
        _fill_gsp_buffer(actor, 200)
        before = T.cat([p.data.flatten().clone() for p in actor.gsp_networks['actor'].parameters()])
        actor.learn_RDDPG(actor.gsp_networks, gsp=True, recurrent=True)
        after = T.cat([p.data.flatten().clone() for p in actor.gsp_networks['actor'].parameters()])
        assert not T.equal(before, after)

    def test_multiple_steps_no_crash(self, config):
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1,
                      gsp=True, recurrent_gsp=True, gsp_input_size=6, gsp_output_size=1,
                      gsp_sequence_length=10, recurrent_hidden_size=32, recurrent_num_layers=2)
        _fill_gsp_buffer(actor, 200)
        for _ in range(20):
            actor.learn_RDDPG(actor.gsp_networks, gsp=True, recurrent=True)

    def test_speed_improvement(self, config):
        """Should be under 200ms per call (was 1500ms)."""
        actor = Actor(id=1, config=config, network="DDPG",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1,
                      gsp=True, recurrent_gsp=True, gsp_input_size=6, gsp_output_size=1,
                      gsp_sequence_length=10, recurrent_hidden_size=32, recurrent_num_layers=2)
        _fill_gsp_buffer(actor, 200)
        # Warm up
        actor.learn_RDDPG(actor.gsp_networks, gsp=True, recurrent=True)
        t0 = time.perf_counter()
        for _ in range(10):
            actor.learn_RDDPG(actor.gsp_networks, gsp=True, recurrent=True)
        elapsed = (time.perf_counter() - t0) / 10 * 1000
        assert elapsed < 200, f"learn_RDDPG took {elapsed:.1f}ms (should be <200ms)"
