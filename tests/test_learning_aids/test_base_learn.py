"""Tests verifying that each RL algorithm updates weights and produces finite loss.

For each algorithm (DQN, DDQN, DDPG, TD3):
1. Weights change after a learn step.
2. Loss returned is finite (not NaN or Inf).
3. Smoke test: 10 consecutive learn steps do not crash.
"""
import math

import numpy as np
import torch as T

from gsp_rl.src.actors.actor import Actor

INPUT_SIZE = 8
OUTPUT_SIZE_DISCRETE = 4
OUTPUT_SIZE_CONTINUOUS = 2

CONFIG = {
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


def make_actor(network: str) -> Actor:
    """Construct an Actor for the given algorithm name."""
    continuous = network in {"DDPG", "TD3"}
    output_size = OUTPUT_SIZE_CONTINUOUS if continuous else OUTPUT_SIZE_DISCRETE
    min_max_action = 1.0 if continuous else 1
    return Actor(
        id=1,
        config=CONFIG,
        network=network,
        input_size=INPUT_SIZE,
        output_size=output_size,
        min_max_action=min_max_action,
        meta_param_size=1,
    )


def fill_buffer(actor: Actor, n: int = 20, continuous: bool = False) -> None:
    """Store n random transitions in the actor's replay buffer."""
    output_size = OUTPUT_SIZE_CONTINUOUS if continuous else OUTPUT_SIZE_DISCRETE
    for _ in range(n):
        s = np.random.randn(INPUT_SIZE).astype(np.float32)
        a = (
            np.random.randn(output_size).astype(np.float32)
            if continuous
            else np.random.randint(0, output_size)
        )
        r = float(np.random.randn())
        s_ = np.random.randn(INPUT_SIZE).astype(np.float32)
        d = bool(np.random.rand() > 0.8)
        actor.store_transition(s, a, r, s_, d, actor.networks)


def get_params_snapshot(network: T.nn.Module) -> T.Tensor:
    """Return a flat tensor of all parameter values."""
    return T.cat([p.data.flatten().clone() for p in network.parameters()])


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

class TestDQN:
    def setup_method(self):
        np.random.seed(0)
        T.manual_seed(0)
        self.actor = make_actor("DQN")
        fill_buffer(self.actor, n=20, continuous=False)

    def test_weights_change_after_learn_step(self):
        before = get_params_snapshot(self.actor.networks["q_eval"])
        self.actor.learn_DQN(self.actor.networks)
        after = get_params_snapshot(self.actor.networks["q_eval"])
        assert not T.allclose(before, after), "DQN q_eval weights did not change after learn step"

    def test_loss_is_finite(self):
        loss = self.actor.learn_DQN(self.actor.networks)
        assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"
        assert math.isfinite(loss), f"DQN loss is not finite: {loss}"

    def test_smoke_10_consecutive_steps(self):
        for _ in range(10):
            loss = self.actor.learn_DQN(self.actor.networks)
            assert math.isfinite(loss), f"DQN loss became non-finite during smoke test: {loss}"


# ---------------------------------------------------------------------------
# DDQN
# ---------------------------------------------------------------------------

class TestDDQN:
    def setup_method(self):
        np.random.seed(0)
        T.manual_seed(0)
        self.actor = make_actor("DDQN")
        fill_buffer(self.actor, n=20, continuous=False)

    def test_weights_change_after_learn_step(self):
        before = get_params_snapshot(self.actor.networks["q_eval"])
        self.actor.learn_DDQN(self.actor.networks)
        after = get_params_snapshot(self.actor.networks["q_eval"])
        assert not T.allclose(before, after), "DDQN q_eval weights did not change after learn step"

    def test_loss_is_finite(self):
        loss = self.actor.learn_DDQN(self.actor.networks)
        assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"
        assert math.isfinite(loss), f"DDQN loss is not finite: {loss}"

    def test_smoke_10_consecutive_steps(self):
        for _ in range(10):
            loss = self.actor.learn_DDQN(self.actor.networks)
            assert math.isfinite(loss), f"DDQN loss became non-finite during smoke test: {loss}"


# ---------------------------------------------------------------------------
# DDPG
# ---------------------------------------------------------------------------

class TestDDPG:
    def setup_method(self):
        np.random.seed(0)
        T.manual_seed(0)
        self.actor = make_actor("DDPG")
        fill_buffer(self.actor, n=20, continuous=True)

    def test_actor_weights_change_after_learn_step(self):
        before = get_params_snapshot(self.actor.networks["actor"])
        self.actor.learn_DDPG(self.actor.networks)
        after = get_params_snapshot(self.actor.networks["actor"])
        assert not T.allclose(before, after), "DDPG actor weights did not change after learn step"

    def test_critic_weights_change_after_learn_step(self):
        before = get_params_snapshot(self.actor.networks["critic"])
        self.actor.learn_DDPG(self.actor.networks)
        after = get_params_snapshot(self.actor.networks["critic"])
        assert not T.allclose(before, after), "DDPG critic weights did not change after learn step"

    def test_loss_is_finite(self):
        loss = self.actor.learn_DDPG(self.actor.networks)
        assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"
        assert math.isfinite(loss), f"DDPG actor_loss is not finite: {loss}"

    def test_smoke_10_consecutive_steps(self):
        for _ in range(10):
            loss = self.actor.learn_DDPG(self.actor.networks)
            assert math.isfinite(loss), f"DDPG loss became non-finite during smoke test: {loss}"


# ---------------------------------------------------------------------------
# TD3
# ---------------------------------------------------------------------------

class TestTD3:
    def setup_method(self):
        np.random.seed(0)
        T.manual_seed(0)
        self.actor = make_actor("TD3")
        fill_buffer(self.actor, n=20, continuous=True)

    def _learn_and_get_actor_loss(self):
        """Call learn_TD3 and return the actor loss as float.

        learn_TD3 returns (0, 0) on critic-only steps and actor_loss.item() on
        actor update steps (controlled by update_actor_iter=1 in CONFIG so
        every step updates the actor).
        """
        result = self.actor.learn_TD3(self.actor.networks)
        # With UPDATE_ACTOR_ITER=1 the actor updates every step and returns a float.
        # Guard against the (0, 0) tuple returned on critic-only steps.
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)

    def test_actor_weights_change_after_learn_step(self):
        before = get_params_snapshot(self.actor.networks["actor"])
        self._learn_and_get_actor_loss()
        after = get_params_snapshot(self.actor.networks["actor"])
        assert not T.allclose(before, after), "TD3 actor weights did not change after learn step"

    def test_critic_1_weights_change_after_learn_step(self):
        before = get_params_snapshot(self.actor.networks["critic_1"])
        self._learn_and_get_actor_loss()
        after = get_params_snapshot(self.actor.networks["critic_1"])
        assert not T.allclose(before, after), "TD3 critic_1 weights did not change after learn step"

    def test_critic_2_weights_change_after_learn_step(self):
        before = get_params_snapshot(self.actor.networks["critic_2"])
        self._learn_and_get_actor_loss()
        after = get_params_snapshot(self.actor.networks["critic_2"])
        assert not T.allclose(before, after), "TD3 critic_2 weights did not change after learn step"

    def test_loss_is_finite(self):
        loss = self._learn_and_get_actor_loss()
        assert math.isfinite(loss), f"TD3 actor_loss is not finite: {loss}"

    def test_smoke_10_consecutive_steps(self):
        for _ in range(10):
            loss = self._learn_and_get_actor_loss()
            assert math.isfinite(loss), f"TD3 loss became non-finite during smoke test: {loss}"
