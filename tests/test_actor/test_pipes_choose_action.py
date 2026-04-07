"""Tests for Actor GSP pipeline construction and action selection.

Covers:
- TestActorWithoutGSP: gsp=False does not build gsp_networks, input_size not augmented.
- TestActorWithGSP: gsp=True builds gsp_networks, input_size augmented.
- TestChooseAction: choose_action returns valid outputs for DQN, DDPG, TD3.

NOTE: test_gsp_false_no_gsp_networks is expected to FAIL with current code due to
the known bug at actor.py line ~110 where `if gsp is not None` is always True
since False is not None. This is a TDD test confirming the bug (fixed in Task 9).
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


def make_actor(network="DQN", gsp=False, gsp_input_size=6, gsp_output_size=1, config=None):
    if config is None:
        config = BASE_CONFIG
    kwargs = dict(
        id=1,
        config=config,
        network=network,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        min_max_action=1,
        meta_param_size=1,
        gsp=gsp,
    )
    if gsp:
        kwargs["gsp_input_size"] = gsp_input_size
        kwargs["gsp_output_size"] = gsp_output_size
    return Actor(**kwargs)


class TestActorWithoutGSP:
    """Actor constructed with gsp=False."""

    def test_gsp_false_no_gsp_networks(self):
        """gsp=False should leave gsp_networks as None.

        KNOWN FAILING TEST (TDD): The current code at actor.py ~line 110 uses
        `if gsp is not None` which is always True since False is not None.
        This causes build_gsp_network('DDPG') to always be called even when
        gsp=False. This test documents the bug and will be fixed in Task 9.
        """
        actor = make_actor(network="DQN", gsp=False)
        assert actor.gsp_networks is None

    def test_input_size_not_augmented_without_gsp(self):
        """Without GSP, network_input_size should equal the raw input_size."""
        actor = make_actor(network="DQN", gsp=False)
        assert actor.network_input_size == INPUT_SIZE


class TestActorWithGSP:
    """Actor constructed with gsp=True."""

    def test_gsp_true_builds_gsp_networks(self):
        """With gsp=True, gsp_networks should be populated with network keys."""
        actor = make_actor(network="DQN", gsp=True, gsp_input_size=6, gsp_output_size=1)
        assert actor.gsp_networks is not None
        assert len(actor.gsp_networks) > 0

    def test_input_size_augmented_with_gsp(self):
        """With gsp=True, network_input_size should be input_size + gsp_output_size."""
        gsp_output_size = 1
        actor = make_actor(network="DQN", gsp=True, gsp_input_size=6, gsp_output_size=gsp_output_size)
        assert actor.network_input_size == INPUT_SIZE + gsp_output_size


class TestChooseAction:
    """Actor.choose_action returns valid outputs for each algorithm."""

    def test_dqn_returns_valid_action(self):
        """DQN choose_action returns an int in range [0, output_size)."""
        actor = make_actor(network="DQN", gsp=False)
        observation = np.random.random(INPUT_SIZE).astype(np.float32)
        action = actor.choose_action(observation, actor.networks, test=True)
        assert isinstance(action, int)
        assert 0 <= action < OUTPUT_SIZE

    def test_ddpg_returns_bounded_action(self):
        """DDPG choose_action returns a numpy array of shape (output_size,).

        actor.py choose_action for DDPG returns actions[0].cpu().detach().numpy()
        where actions has shape (1, output_size), so the returned array is (output_size,).
        """
        actor = make_actor(network="DDPG", gsp=False)
        observation = np.random.random(INPUT_SIZE).astype(np.float32)
        action = actor.choose_action(observation, actor.networks, test=True)
        assert hasattr(action, "shape")
        assert action.shape == (OUTPUT_SIZE,)

    def test_td3_returns_bounded_action(self):
        """TD3 choose_action returns a numpy array of shape (output_size,).

        actor.py choose_action for TD3 returns actions[0] where actions has
        shape (1, output_size) from TD3_choose_action, giving shape (output_size,).
        """
        actor = make_actor(network="TD3", gsp=False)
        observation = np.random.random(INPUT_SIZE).astype(np.float32)
        action = actor.choose_action(observation, actor.networks, test=True)
        assert hasattr(action, "shape")
        assert action.shape == (OUTPUT_SIZE,)
