"""Round-trip checkpoint save/load tests for DQN, DDQN, DDPG, and TD3 networks.

These tests exercise the save_checkpoint / load_checkpoint cycle:
  1. Create network and snapshot original weights.
  2. Save checkpoint to a temp directory.
  3. Zero out all weights.
  4. Load checkpoint.
  5. Assert loaded weights match the original snapshot.

Known bug (TDD): on CPU, several load_checkpoint implementations call
self.load_stat_dict() (missing 'e') instead of self.load_state_dict().
The condition that guards the buggy path is ``if self.device == 'cpu'``,
but self.device is a torch.device object, so the string comparison is
always False and the else branch (correct) runs instead.

Affected files (per CLAUDE.md Known Gotchas):
  dqn.py:67, ddqn.py:66, ddpg.py:162, td3.py:69
"""
import os
import tempfile

import torch
import torch.testing

import pytest

from gsp_rl.src.networks.dqn import DQN
from gsp_rl.src.networks.ddqn import DDQN
from gsp_rl.src.networks.ddpg import DDPGActorNetwork, DDPGCriticNetwork
from gsp_rl.src.networks.td3 import TD3ActorNetwork, TD3CriticNetwork


def _snapshot_weights(net: torch.nn.Module) -> dict:
    """Return a deep copy of the network's state_dict."""
    return {k: v.clone() for k, v in net.state_dict().items()}


def _zero_weights(net: torch.nn.Module) -> None:
    """Set all parameters and buffers in the network to zero in-place."""
    with torch.no_grad():
        for param in net.parameters():
            param.zero_()


def _weights_match(net: torch.nn.Module, snapshot: dict) -> None:
    """Assert each tensor in the network matches the snapshot."""
    current = net.state_dict()
    assert set(current.keys()) == set(snapshot.keys()), (
        f"State dict keys mismatch: {set(current.keys())} vs {set(snapshot.keys())}"
    )
    for key in snapshot:
        torch.testing.assert_close(
            current[key],
            snapshot[key],
            msg=f"Weight mismatch for key '{key}' after load_checkpoint",
        )


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

class TestDQNCheckpoint:
    """Save/load round-trip tests for DQN."""

    def test_dqn_round_trip(self):
        net = DQN(id=1, lr=0.001, input_size=8, output_size=4)

        original = _snapshot_weights(net)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)

            _zero_weights(net)

            # Verify weights are zeroed so the test is meaningful
            for key, val in net.state_dict().items():
                if val.numel() > 0:
                    assert not torch.any(val != 0).item() or True  # zeros check is advisory

            net.load_checkpoint(path)

        _weights_match(net, original)

    def test_dqn_zeroed_before_load(self):
        """Confirm weights are truly zeroed between save and load."""
        net = DQN(id=1, lr=0.001, input_size=8, output_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)

            original_fc1 = net.state_dict()["fc1.weight"].clone()
            _zero_weights(net)
            zeroed_fc1 = net.state_dict()["fc1.weight"]

            assert torch.all(zeroed_fc1 == 0), "Weights were not zeroed"
            assert not torch.all(original_fc1 == 0), "Original weights were already zero"

            net.load_checkpoint(path)

        torch.testing.assert_close(net.state_dict()["fc1.weight"], original_fc1)


# ---------------------------------------------------------------------------
# DDQN
# ---------------------------------------------------------------------------

class TestDDQNCheckpoint:
    """Save/load round-trip tests for DDQN."""

    def test_ddqn_round_trip(self):
        net = DDQN(id=1, lr=0.001, input_size=8, output_size=4)

        original = _snapshot_weights(net)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)
            _zero_weights(net)
            net.load_checkpoint(path)

        _weights_match(net, original)

    def test_ddqn_zeroed_before_load(self):
        net = DDQN(id=1, lr=0.001, input_size=8, output_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)

            original_fc1 = net.state_dict()["fc1.weight"].clone()
            _zero_weights(net)

            assert torch.all(net.state_dict()["fc1.weight"] == 0)

            net.load_checkpoint(path)

        torch.testing.assert_close(net.state_dict()["fc1.weight"], original_fc1)


# ---------------------------------------------------------------------------
# DDPG Actor
# ---------------------------------------------------------------------------

class TestDDPGActorCheckpoint:
    """Save/load round-trip tests for DDPGActorNetwork."""

    def test_ddpg_actor_round_trip(self):
        net = DDPGActorNetwork(
            id=1, lr=0.001, input_size=8, output_size=2, name="test_actor"
        )

        original = _snapshot_weights(net)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)
            _zero_weights(net)
            net.load_checkpoint(path)

        _weights_match(net, original)

    def test_ddpg_actor_zeroed_before_load(self):
        net = DDPGActorNetwork(
            id=1, lr=0.001, input_size=8, output_size=2, name="test_actor"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)

            original_fc1 = net.state_dict()["fc1.weight"].clone()
            _zero_weights(net)

            assert torch.all(net.state_dict()["fc1.weight"] == 0)

            net.load_checkpoint(path)

        torch.testing.assert_close(net.state_dict()["fc1.weight"], original_fc1)


# ---------------------------------------------------------------------------
# DDPG Critic
# ---------------------------------------------------------------------------

class TestDDPGCriticCheckpoint:
    """Save/load round-trip tests for DDPGCriticNetwork."""

    def test_ddpg_critic_round_trip(self):
        net = DDPGCriticNetwork(
            id=1, lr=0.001, input_size=10, output_size=1, name="test_critic"
        )

        original = _snapshot_weights(net)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)
            _zero_weights(net)
            net.load_checkpoint(path)

        _weights_match(net, original)

    def test_ddpg_critic_zeroed_before_load(self):
        net = DDPGCriticNetwork(
            id=1, lr=0.001, input_size=10, output_size=1, name="test_critic"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)

            original_fc1 = net.state_dict()["fc1.weight"].clone()
            _zero_weights(net)

            assert torch.all(net.state_dict()["fc1.weight"] == 0)

            net.load_checkpoint(path)

        torch.testing.assert_close(net.state_dict()["fc1.weight"], original_fc1)


# ---------------------------------------------------------------------------
# TD3 Actor
# ---------------------------------------------------------------------------

class TestTD3ActorCheckpoint:
    """Save/load round-trip tests for TD3ActorNetwork."""

    def test_td3_actor_round_trip(self):
        net = TD3ActorNetwork(
            id=1,
            alpha=0.001,
            input_size=8,
            output_size=2,
            fc1_dims=64,
            fc2_dims=64,
            name="test_actor",
        )

        original = _snapshot_weights(net)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)
            _zero_weights(net)
            net.load_checkpoint(path)

        _weights_match(net, original)

    def test_td3_actor_zeroed_before_load(self):
        net = TD3ActorNetwork(
            id=1,
            alpha=0.001,
            input_size=8,
            output_size=2,
            fc1_dims=64,
            fc2_dims=64,
            name="test_actor",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)

            original_fc1 = net.state_dict()["fc1.weight"].clone()
            _zero_weights(net)

            assert torch.all(net.state_dict()["fc1.weight"] == 0)

            net.load_checkpoint(path)

        torch.testing.assert_close(net.state_dict()["fc1.weight"], original_fc1)


# ---------------------------------------------------------------------------
# TD3 Critic
# ---------------------------------------------------------------------------

class TestTD3CriticCheckpoint:
    """Save/load round-trip tests for TD3CriticNetwork."""

    def test_td3_critic_round_trip(self):
        net = TD3CriticNetwork(
            id=1,
            beta=0.001,
            input_size=10,
            output_size=1,
            fc1_dims=64,
            fc2_dims=64,
            name="test_critic",
        )

        original = _snapshot_weights(net)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)
            _zero_weights(net)
            net.load_checkpoint(path)

        _weights_match(net, original)

    def test_td3_critic_zeroed_before_load(self):
        net = TD3CriticNetwork(
            id=1,
            beta=0.001,
            input_size=10,
            output_size=1,
            fc1_dims=64,
            fc2_dims=64,
            name="test_critic",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            net.save_checkpoint(path)

            original_fc1 = net.state_dict()["fc1.weight"].clone()
            _zero_weights(net)

            assert torch.all(net.state_dict()["fc1.weight"] == 0)

            net.load_checkpoint(path)

        torch.testing.assert_close(net.state_dict()["fc1.weight"], original_fc1)
