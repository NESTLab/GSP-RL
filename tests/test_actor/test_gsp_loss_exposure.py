"""Tests for exposing per-step GSP prediction loss from Actor.learn().

Context: the information-collapse diagnostic (see Stelaris
docs/specs/2026-04-12-dispatcher-diagnostic-batch.md) requires logging the
GSP prediction network's training loss per learning step. The primary loss
already returned from learn() is the actor/critic loss, which stays normal
even when the GSP prediction head has collapsed to a near-constant output.
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
    # Fire GSP learning every primary learn() call so the test doesn't need to iterate 100 times.
    "GSP_LEARNING_FREQUENCY": 1,
    "GSP_BATCH_SIZE": 8,
}

INPUT_SIZE = 8
OUTPUT_SIZE = 4
GSP_INPUT_SIZE = 6
GSP_OUTPUT_SIZE = 1
N_TRANSITIONS = 30


def make_gsp_actor(network="DDPG"):
    return Actor(
        id=1,
        config=BASE_CONFIG,
        network=network,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        min_max_action=1,
        meta_param_size=1,
        gsp=True,
        gsp_input_size=GSP_INPUT_SIZE,
        gsp_output_size=GSP_OUTPUT_SIZE,
    )


def fill_primary_and_gsp_buffers(actor, n=N_TRANSITIONS):
    for _ in range(n):
        # Primary network sees inputs that include the GSP head output (network_input_size)
        s = np.random.random(actor.network_input_size).astype(np.float32)
        s_ = np.random.random(actor.network_input_size).astype(np.float32)
        a = actor.choose_action(s, actor.networks, test=True)
        r = float(np.random.random())
        actor.store_transition(s, a, r, s_, False, actor.networks)

        # GSP prediction network training transitions
        gsp_s = np.random.random(GSP_INPUT_SIZE).astype(np.float32)
        gsp_s_ = np.random.random(GSP_INPUT_SIZE).astype(np.float32)
        gsp_a = np.random.uniform(-1, 1, size=GSP_OUTPUT_SIZE).astype(np.float32)
        gsp_r = float(np.random.random())
        actor.store_gsp_transition(gsp_s, gsp_a, gsp_r, gsp_s_, False)


class TestGSPLossExposure:
    def test_last_gsp_loss_initialized_to_none(self):
        actor = make_gsp_actor()
        assert actor.last_gsp_loss is None

    def test_last_gsp_loss_populated_after_learn_with_gsp(self):
        """After primary + GSP buffers are filled and learn() runs, last_gsp_loss is a float."""
        actor = make_gsp_actor()
        fill_primary_and_gsp_buffers(actor)
        actor.learn()
        assert actor.last_gsp_loss is not None
        assert isinstance(actor.last_gsp_loss, float)

    def test_last_gsp_loss_resets_between_ticks(self):
        """Each learn() call starts by resetting last_gsp_loss to None.

        This is the load-bearing invariant of the field: consumers must be able to read
        it after learn() and distinguish "no GSP step ran this tick" (None) from
        "GSP step ran and returned a value" (float). If the reset fails, a stale value
        from a previous tick bleeds into the current tick's reading.
        """
        actor = make_gsp_actor()
        fill_primary_and_gsp_buffers(actor)
        actor.learn()
        assert actor.last_gsp_loss is not None  # populated after first learn

        # Drain the GSP replay buffer below the batch size so the next learn_gsp early-returns.
        # Simulate by swapping in an empty gsp replay buffer. This is a white-box probe of the
        # reset invariant rather than a full end-to-end run.
        actor.gsp_networks['replay'].mem_ctr = 0

        actor.learn()
        assert actor.last_gsp_loss is None, (
            "last_gsp_loss should reset to None when learn() runs but no GSP step fires"
        )

    def test_last_gsp_loss_remains_none_when_gsp_disabled(self):
        """Non-GSP actor never populates last_gsp_loss."""
        actor = Actor(
            id=1,
            config=BASE_CONFIG,
            network="DDPG",
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            min_max_action=1,
            meta_param_size=1,
            gsp=False,
        )
        for _ in range(N_TRANSITIONS):
            s = np.random.random(INPUT_SIZE).astype(np.float32)
            a = actor.choose_action(s, actor.networks, test=True)
            r = float(np.random.random())
            s_ = np.random.random(INPUT_SIZE).astype(np.float32)
            actor.store_transition(s, a, r, s_, False, actor.networks)
        actor.learn()
        assert actor.last_gsp_loss is None
