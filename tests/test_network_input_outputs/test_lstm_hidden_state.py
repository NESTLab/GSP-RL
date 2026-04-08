"""Tests for EnvironmentEncoder hidden state management."""

import torch as T
import numpy as np
import pytest

from gsp_rl.src.networks.lstm import EnvironmentEncoder


@pytest.fixture
def encoder():
    return EnvironmentEncoder(
        input_size=6, output_size=1, hidden_size=32,
        embedding_size=32, batch_size=8, num_layers=2, lr=0.001
    )


class TestHiddenStateAPI:
    def test_forward_returns_output_and_hidden(self, encoder):
        x = T.randn(5, 6).to(encoder.device)
        result = encoder(x)
        assert isinstance(result, tuple)
        assert len(result) == 2
        output, (h_n, c_n) = result
        assert output.shape == (5, 1)
        assert h_n.shape == (2, 1, 32)  # (layers, batch=1, hidden)
        assert c_n.shape == (2, 1, 32)

    def test_forward_with_hidden_differs_from_zeros(self, encoder):
        x = T.randn(5, 6).to(encoder.device)
        out_zero, _ = encoder(x)
        h_0 = T.randn(2, 1, 32).to(encoder.device)
        c_0 = T.randn(2, 1, 32).to(encoder.device)
        out_hidden, _ = encoder(x, hidden=(h_0, c_0))
        assert not T.allclose(out_zero, out_hidden)

    def test_hidden_carries_across_calls(self, encoder):
        x1 = T.randn(5, 6).to(encoder.device)
        x2 = T.randn(5, 6).to(encoder.device)
        _, (h1, c1) = encoder(x1)
        out_carried, _ = encoder(x2, hidden=(h1, c1))
        out_fresh, _ = encoder(x2)
        assert not T.allclose(out_carried, out_fresh)

    def test_batch_forward(self, encoder):
        x = T.randn(4, 5, 6).to(encoder.device)
        output, (h_n, c_n) = encoder(x)
        assert output.shape == (4, 5, 1)
        assert h_n.shape[1] == 4

    def test_batch_with_hidden(self, encoder):
        x = T.randn(4, 5, 6).to(encoder.device)
        h_0 = T.randn(2, 4, 32).to(encoder.device)
        c_0 = T.randn(2, 4, 32).to(encoder.device)
        output, (h_n, c_n) = encoder(x, hidden=(h_0, c_0))
        assert output.shape == (4, 5, 1)
        assert h_n.shape == (2, 4, 32)

    def test_backward_works(self, encoder):
        x = T.randn(5, 6).to(encoder.device)
        output, _ = encoder(x)
        loss = output.sum()
        loss.backward()

    def test_backward_with_hidden(self, encoder):
        x = T.randn(4, 5, 6).to(encoder.device)
        h_0 = T.randn(2, 4, 32).to(encoder.device)
        c_0 = T.randn(2, 4, 32).to(encoder.device)
        output, _ = encoder(x, hidden=(h_0, c_0))
        loss = output.sum()
        loss.backward()
