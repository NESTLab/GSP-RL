"""Tests for direct-MSE GSP training (Option A from the collapse analysis).

Verifies that:
1. After training on a deterministic state→label mapping, the GSP predictor's
   MSE is lower than a trivial "predict the mean" baseline.
2. `last_gsp_loss` is populated after learn() runs for a DDPG-GSP actor.

See docs/research/2026-04-13-gsp-information-collapse-analysis.md in Stelaris
for the root cause and the rationale for switching from DDPG actor-critic to
direct supervised MSE.
"""

import numpy as np
import pytest
import torch

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
    "BATCH_SIZE": 16,
    "MEM_SIZE": 1000,
    "REPLACE_TARGET_COUNTER": 10,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 1,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 1,
    "GSP_BATCH_SIZE": 16,
}


INPUT_SIZE = 8
OUTPUT_SIZE = 4
GSP_INPUT_SIZE = 6
GSP_OUTPUT_SIZE = 1


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


def _fill_gsp_buffer_with_linear_labels(actor, n_transitions=400, seed=0):
    """Store (state, label) pairs where label = mean(state).

    A predictor trained with direct MSE should beat the trivial-mean baseline
    within ~200 steps on this trivially-learnable mapping.
    """
    rng = np.random.default_rng(seed)
    stored_states = []
    stored_labels = []
    for _ in range(n_transitions):
        state = rng.uniform(-1, 1, size=GSP_INPUT_SIZE).astype(np.float32)
        label = float(np.mean(state))
        # Label carried in the action field under the new direct-MSE convention.
        actor.store_gsp_transition(
            state, np.float32(label), 0.0, np.zeros_like(state), False
        )
        stored_states.append(state)
        stored_labels.append(label)
    return np.stack(stored_states), np.array(stored_labels, dtype=np.float32)


def _fill_primary_buffer(actor, n=20):
    """Primary replay buffer must have >= BATCH_SIZE transitions for learn()."""
    rng = np.random.default_rng(42)
    for _ in range(n):
        s = rng.random(actor.network_input_size).astype(np.float32)
        s_ = rng.random(actor.network_input_size).astype(np.float32)
        a = actor.choose_action(s, actor.networks, test=True)
        actor.store_transition(s, a, 0.0, s_, False, actor.networks)


def test_learn_gsp_mse_beats_trivial_mean_baseline_on_linear_task():
    """After training, the predictor's MSE is lower than predicting the constant mean."""
    torch.manual_seed(0)
    np.random.seed(0)
    actor = make_gsp_actor(network="DDPG")
    states, labels = _fill_gsp_buffer_with_linear_labels(actor, n_transitions=400, seed=0)
    _fill_primary_buffer(actor)

    for _ in range(200):
        actor.learn()

    net = actor.gsp_networks["actor"]
    with torch.no_grad():
        states_t = torch.from_numpy(states).to(net.device)
        preds = net.forward(states_t).cpu().numpy().ravel()

    pred_mse = float(np.mean((preds - labels) ** 2))
    trivial_mse = float(np.mean((labels - labels.mean()) ** 2))
    assert pred_mse < trivial_mse, (
        f"Direct-MSE GSP training did not beat trivial baseline: "
        f"pred_mse={pred_mse:.5f} trivial_mse={trivial_mse:.5f}"
    )


def test_learn_gsp_populates_last_gsp_loss_for_ddpg_variant():
    """last_gsp_loss is populated after learn() for a DDPG-GSP actor."""
    torch.manual_seed(0)
    actor = make_gsp_actor(network="DDPG")
    _fill_gsp_buffer_with_linear_labels(actor, n_transitions=100, seed=1)
    _fill_primary_buffer(actor)

    actor.learn()
    assert actor.last_gsp_loss is not None
    assert isinstance(actor.last_gsp_loss, float)
    assert actor.last_gsp_loss >= 0.0
