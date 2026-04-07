"""Tests for TD3-specific correctness: delayed updates, gradient isolation, initialization.

These tests verify the three core TD3 innovations (Fujimoto et al. 2018):
1. Clipped Double-Q Learning (twin critics, min for target)
2. Delayed Policy Updates (actor + targets updated every d steps)
3. Target Policy Smoothing (noise added to target actions)
"""

import numpy as np
import torch as T
import pytest

from gsp_rl.src.actors.actor import Actor


@pytest.fixture
def config():
    return {
        "GAMMA": 0.99, "TAU": 0.5, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 1000, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.1, "UPDATE_ACTOR_ITER": 2, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }


def _fill_buffer(actor, n=20):
    for _ in range(n):
        s = np.random.randn(8).astype(np.float32)
        a = np.random.randn(2).astype(np.float32)
        r = np.random.randn()
        s_ = np.random.randn(8).astype(np.float32)
        d = bool(np.random.rand() > 0.8)
        actor.store_transition(s, a, r, s_, d, actor.networks)


def _get_params_snapshot(network):
    return {name: p.data.clone() for name, p in network.named_parameters()}


class TestDelayedPolicyUpdate:
    """TD3 should only update the actor every UPDATE_ACTOR_ITER steps."""

    def test_critic_updates_every_step(self, config):
        """Critics should update on every learn step."""
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        _fill_buffer(actor, 30)

        critic_before = _get_params_snapshot(actor.networks["critic_1"])
        actor.learn()
        critic_after = _get_params_snapshot(actor.networks["critic_1"])

        changed = any(
            not T.equal(critic_before[n], critic_after[n])
            for n in critic_before
        )
        assert changed, "Critic should update on every learn step"

    def test_actor_does_not_update_on_non_delayed_step(self, config):
        """Actor should NOT update when learn_step_counter % UPDATE_ACTOR_ITER != 0.
        Note: learn_TD3 increments counter BEFORE checking, so to get a critic-only
        step (counter after increment is odd), set counter to 0 (becomes 1, 1%2!=0)."""
        config["UPDATE_ACTOR_ITER"] = 2
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        _fill_buffer(actor, 30)

        # Counter=0 -> incremented to 1 -> 1%2!=0 -> critic-only step
        actor.networks["learn_step_counter"] = 0

        actor_before = _get_params_snapshot(actor.networks["actor"])
        actor.learn()
        actor_after = _get_params_snapshot(actor.networks["actor"])

        unchanged = all(
            T.equal(actor_before[n], actor_after[n])
            for n in actor_before
        )
        assert unchanged, "Actor should NOT update on critic-only steps"

    def test_actor_updates_on_delayed_step(self, config):
        """Actor SHOULD update when learn_step_counter % UPDATE_ACTOR_ITER == 0.
        Note: learn_TD3 increments counter BEFORE checking, so to get an actor
        update step (counter after increment is even), set counter to 1 (becomes 2, 2%2==0)."""
        config["UPDATE_ACTOR_ITER"] = 2
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        _fill_buffer(actor, 30)

        # Counter=1 -> incremented to 2 -> 2%2==0 -> actor update step
        actor.networks["learn_step_counter"] = 1

        actor_before = _get_params_snapshot(actor.networks["actor"])
        actor.learn()
        actor_after = _get_params_snapshot(actor.networks["actor"])

        changed = any(
            not T.equal(actor_before[n], actor_after[n])
            for n in actor_before
        )
        assert changed, "Actor should update on delayed update steps"

    def test_target_only_updates_on_actor_update_step(self, config):
        """Target networks should only update when the actor updates (delayed).
        Counter=0 -> incremented to 1 -> critic-only -> no target update."""
        config["UPDATE_ACTOR_ITER"] = 2
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        _fill_buffer(actor, 30)

        # Counter=0 -> incremented to 1 -> 1%2!=0 -> critic-only, no target update
        actor.networks["learn_step_counter"] = 0
        target_before = _get_params_snapshot(actor.networks["target_actor"])
        actor.learn()
        target_after = _get_params_snapshot(actor.networks["target_actor"])

        unchanged = all(
            T.equal(target_before[n], target_after[n])
            for n in target_before
        )
        assert unchanged, (
            "Target actor should NOT update on critic-only steps. "
            "This is a core TD3 innovation — delayed target updates."
        )


class TestGradientIsolation:
    """Target network computations should not track gradients."""

    def test_target_networks_have_no_grad_after_learn(self, config):
        """After a learn step, target network params should have no gradients."""
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        _fill_buffer(actor, 30)
        actor.learn()

        for name, p in actor.networks["target_actor"].named_parameters():
            assert p.grad is None, (
                f"Target actor param {name} has gradient — "
                f"target computation should use T.no_grad()"
            )
        for name, p in actor.networks["target_critic_1"].named_parameters():
            assert p.grad is None, (
                f"Target critic_1 param {name} has gradient — "
                f"target computation should use T.no_grad()"
            )
        for name, p in actor.networks["target_critic_2"].named_parameters():
            assert p.grad is None, (
                f"Target critic_2 param {name} has gradient — "
                f"target computation should use T.no_grad()"
            )


class TestTargetPolicySmoothing:
    """Target actions should have per-dimension noise, not scalar noise."""

    def test_multi_dim_action_learning_does_not_crash(self, config):
        """TD3 with multi-dimensional actions should train without error."""
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=4, min_max_action=1.0, meta_param_size=1)
        # Fill buffer with correct action dimensionality
        for _ in range(30):
            s = np.random.randn(8).astype(np.float32)
            a = np.random.randn(4).astype(np.float32)
            r = np.random.randn()
            s_ = np.random.randn(8).astype(np.float32)
            d = bool(np.random.rand() > 0.8)
            actor.store_transition(s, a, r, s_, d, actor.networks)

        # Should not crash — verifies noise shape matches action dims
        for _ in range(5):
            actor.learn()


class TestWeightInitialization:
    """TD3 should use fanin_init like DDPG for stable early learning."""

    def test_td3_actor_output_layer_has_small_weights(self, config):
        """Output layer weights should be small (like DDPG's uniform[-3e-3, 3e-3])."""
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)

        output_weights = None
        for name, p in actor.networks["actor"].named_parameters():
            if "fc3" in name and "weight" in name:
                output_weights = p.data
                break
            # TD3 might use different layer names — find the last linear weight
            if "weight" in name:
                output_weights = p.data

        assert output_weights is not None
        max_val = output_weights.abs().max().item()
        # With fanin_init, output weights should be bounded small (~0.003)
        # With default Kaiming init, they can be much larger (~0.5)
        assert max_val < 0.1, (
            f"TD3 actor output weights max={max_val:.4f} — too large. "
            f"Should use fanin_init like DDPG for stable early learning."
        )


class TestUpdateOrder:
    """Soft target update should happen AFTER learning, not before."""

    def test_learn_then_update_not_update_then_learn(self, config):
        """After Actor.learn(), the target should reflect the CURRENT online params,
        not the previous step's params."""
        config["TAU"] = 1.0  # tau=1 means target = online after update
        config["UPDATE_ACTOR_ITER"] = 1  # update every step
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        _fill_buffer(actor, 30)

        actor.learn()

        # With tau=1 and correct order (learn then update):
        # target params should equal current online params
        for (name_online, p_online), (name_target, p_target) in zip(
            actor.networks["actor"].named_parameters(),
            actor.networks["target_actor"].named_parameters()
        ):
            T.testing.assert_close(
                p_online.data, p_target.data, atol=1e-5, rtol=1e-5,
                msg=f"With tau=1, target {name_target} should equal online {name_online} "
                    f"after learn+update. If they differ, update happens before learn."
            )
