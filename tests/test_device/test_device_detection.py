"""Tests for automatic device detection including Apple Silicon MPS.

Device priority: cuda > mps > cpu
All networks should use the best available device automatically.
Forward + backward passes should work on the detected device.
"""

import numpy as np
import torch as T
import pytest

from gsp_rl.src.networks.dqn import DQN
from gsp_rl.src.networks.ddqn import DDQN
from gsp_rl.src.networks.ddpg import DDPGActorNetwork, DDPGCriticNetwork
from gsp_rl.src.networks.td3 import TD3ActorNetwork, TD3CriticNetwork
from gsp_rl.src.networks.lstm import EnvironmentEncoder
from gsp_rl.src.networks.self_attention import AttentionEncoder


def _expected_device(recurrent=False):
    """Return the device that get_device() should detect."""
    if T.cuda.is_available():
        return "cuda"
    elif T.backends.mps.is_available():
        if recurrent:
            return "cpu"  # MPS fallback for LSTM/attention
        return "mps"
    else:
        return "cpu"


class TestGetDevice:
    """Test the centralized device detection utility."""

    def test_get_device_exists(self):
        """get_device should be importable from the networks package."""
        from gsp_rl.src.networks import get_device
        device = get_device()
        assert isinstance(device, T.device)

    def test_get_device_returns_best_available(self):
        from gsp_rl.src.networks import get_device
        device = get_device()
        expected = _expected_device()
        assert device.type == expected, (
            f"get_device() returned {device}, expected {expected}. "
            f"cuda={T.cuda.is_available()}, mps={T.backends.mps.is_available()}"
        )

    def test_get_device_recurrent_fallback(self):
        """Recurrent networks should fall back to CPU on MPS (not on CUDA)."""
        from gsp_rl.src.networks import get_device
        device = get_device(recurrent=True)
        expected = _expected_device(recurrent=True)
        assert device.type == expected, (
            f"get_device(recurrent=True) returned {device}, expected {expected}"
        )

    def test_get_device_is_consistent(self):
        """Multiple calls should return the same device."""
        from gsp_rl.src.networks import get_device
        d1 = get_device()
        d2 = get_device()
        assert d1 == d2


class TestNetworksUseDetectedDevice:
    """Every network should use the auto-detected device, not hardcoded cuda/cpu."""

    def test_dqn_device(self):
        net = DQN(id=1, lr=0.001, input_size=8, output_size=4)
        assert net.device.type == _expected_device()

    def test_ddqn_device(self):
        net = DDQN(id=1, lr=0.001, input_size=8, output_size=4)
        assert net.device.type == _expected_device()

    def test_ddpg_actor_device(self):
        net = DDPGActorNetwork(id=1, lr=0.001, input_size=8, output_size=2, name="test")
        assert net.device.type == _expected_device()

    def test_ddpg_critic_device(self):
        net = DDPGCriticNetwork(id=1, lr=0.001, input_size=10, output_size=1, name="test")
        assert net.device.type == _expected_device()

    def test_td3_actor_device(self):
        net = TD3ActorNetwork(id=1, alpha=0.001, input_size=8, output_size=2,
                              fc1_dims=64, fc2_dims=64, name="test")
        assert net.device.type == _expected_device()

    def test_td3_critic_device(self):
        net = TD3CriticNetwork(id=1, beta=0.001, input_size=10, output_size=1,
                               fc1_dims=64, fc2_dims=64, name="test")
        assert net.device.type == _expected_device()

    def test_lstm_device(self):
        net = EnvironmentEncoder(input_size=5, output_size=1, hidden_size=32,
                                 embedding_size=32, batch_size=8, num_layers=1, lr=0.001)
        assert net.device.type == _expected_device(recurrent=True)

    def test_attention_device(self):
        net = AttentionEncoder(input_size=5, output_size=1, min_max_action=1.0,
                               encode_size=16, embed_size=16, hidden_size=16,
                               heads=2, forward_expansion=2, dropout=0.0, max_length=5)
        assert net.device.type == _expected_device(recurrent=True)


class TestForwardBackwardOnDevice:
    """Forward and backward passes should work on the detected device."""

    def test_dqn_forward_backward(self):
        net = DQN(id=1, lr=0.001, input_size=8, output_size=4)
        x = T.randn(1, 8).to(net.device)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert out.device.type == net.device.type

    def test_ddqn_forward_backward(self):
        net = DDQN(id=1, lr=0.001, input_size=8, output_size=4)
        x = T.randn(1, 8).to(net.device)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert out.device.type == net.device.type

    def test_ddpg_actor_forward_backward(self):
        net = DDPGActorNetwork(id=1, lr=0.001, input_size=8, output_size=2, name="test")
        x = T.randn(1, 8).to(net.device)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert out.device.type == net.device.type

    def test_ddpg_critic_forward_backward(self):
        net = DDPGCriticNetwork(id=1, lr=0.001, input_size=10, output_size=1, name="test")
        s = T.randn(1, 8).to(net.device)
        a = T.randn(1, 2).to(net.device)
        out = net(s, a)
        loss = out.sum()
        loss.backward()
        assert out.device.type == net.device.type

    def test_td3_actor_forward_backward(self):
        net = TD3ActorNetwork(id=1, alpha=0.001, input_size=8, output_size=2,
                              fc1_dims=64, fc2_dims=64, name="test")
        x = T.randn(1, 8).to(net.device)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert out.device.type == net.device.type

    def test_td3_critic_forward_backward(self):
        net = TD3CriticNetwork(id=1, beta=0.001, input_size=10, output_size=1,
                               fc1_dims=64, fc2_dims=64, name="test")
        s = T.randn(1, 8).to(net.device)
        a = T.randn(1, 2).to(net.device)
        out = net(s, a)
        loss = out.sum()
        loss.backward()
        assert out.device.type == net.device.type

    def test_attention_forward_backward(self):
        net = AttentionEncoder(input_size=5, output_size=1, min_max_action=1.0,
                               encode_size=16, embed_size=16, hidden_size=16,
                               heads=2, forward_expansion=2, dropout=0.0, max_length=5)
        x = T.randn(1, 5, 5).to(net.device)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert out.device.type == net.device.type


class TestFullLearnStepOnDevice:
    """A complete learn step (sample + forward + loss + backward + step) should work."""

    def test_dqn_learn_on_device(self):
        from gsp_rl.src.actors.actor import Actor
        config = {
            "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
            "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
            "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
            "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
            "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
        }
        actor = Actor(id=1, config=config, network="DQN",
                      input_size=8, output_size=4, min_max_action=1, meta_param_size=1)
        for _ in range(20):
            s = np.random.randn(8).astype(np.float32)
            actor.store_transition(s, np.random.randint(0, 4), np.random.randn(),
                                   np.random.randn(8).astype(np.float32), False, actor.networks)
        loss = actor.learn_DQN(actor.networks)
        assert np.isfinite(loss)

    def test_td3_learn_on_device(self):
        from gsp_rl.src.actors.actor import Actor
        config = {
            "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
            "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
            "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
            "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
            "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
        }
        actor = Actor(id=1, config=config, network="TD3",
                      input_size=8, output_size=2, min_max_action=1.0, meta_param_size=1)
        for _ in range(20):
            s = np.random.randn(8).astype(np.float32)
            actor.store_transition(s, np.random.randn(2).astype(np.float32), np.random.randn(),
                                   np.random.randn(8).astype(np.float32), False, actor.networks)
        result = actor.learn_TD3(actor.networks)
        if isinstance(result, tuple):
            assert all(np.isfinite(v) for v in result)
        else:
            assert np.isfinite(result)
