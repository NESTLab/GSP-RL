"""Tests for SequenceReplayBuffer with hidden state storage."""

import numpy as np
import pytest
from gsp_rl.src.buffers.sequential import SequenceReplayBuffer


class TestHiddenStateStorage:
    def test_init_with_hidden(self):
        buf = SequenceReplayBuffer(
            max_sequence=5, num_observations=4, num_actions=2,
            seq_len=3, hidden_size=32, num_layers=2
        )
        assert buf._has_hidden is True
        assert buf.h_memory.shape == (5, 2, 1, 32)

    def test_init_without_hidden_backward_compat(self):
        buf = SequenceReplayBuffer(
            max_sequence=5, num_observations=4, num_actions=2, seq_len=3
        )
        assert buf._has_hidden is False

    def test_set_and_flush_hidden(self):
        buf = SequenceReplayBuffer(
            max_sequence=5, num_observations=4, num_actions=2,
            seq_len=3, hidden_size=32, num_layers=2
        )
        h = np.ones((2, 1, 32), dtype=np.float32) * 42
        c = np.ones((2, 1, 32), dtype=np.float32) * 7
        buf.set_sequence_hidden(h, c)
        for i in range(3):
            buf.store_transition(np.ones(4)*i, np.ones(2), float(i), np.ones(4), False)
        assert buf.mem_ctr == 3
        np.testing.assert_array_equal(buf.h_memory[0], h)
        np.testing.assert_array_equal(buf.c_memory[0], c)

    def test_sample_returns_7_with_hidden(self):
        buf = SequenceReplayBuffer(
            max_sequence=5, num_observations=4, num_actions=2,
            seq_len=3, hidden_size=32, num_layers=2
        )
        for seq in range(3):
            h = np.ones((2, 1, 32), dtype=np.float32) * seq
            c = np.ones((2, 1, 32), dtype=np.float32) * seq
            buf.set_sequence_hidden(h, c)
            for i in range(3):
                buf.store_transition(np.ones(4), np.ones(2), 0.0, np.ones(4), False)
        result = buf.sample_buffer(2)
        assert len(result) == 7
        s, a, r, s_, d, h_batch, c_batch = result
        assert h_batch.shape == (2, 2, 1, 32)
        assert c_batch.shape == (2, 2, 1, 32)

    def test_sample_returns_5_without_hidden(self):
        buf = SequenceReplayBuffer(
            max_sequence=5, num_observations=4, num_actions=2, seq_len=3
        )
        for i in range(6):
            buf.store_transition(np.ones(4), np.ones(2), 0.0, np.ones(4), False)
        result = buf.sample_buffer(1)
        assert len(result) == 5

    def test_hidden_values_match_stored(self):
        buf = SequenceReplayBuffer(
            max_sequence=10, num_observations=4, num_actions=2,
            seq_len=3, hidden_size=16, num_layers=1
        )
        # Store 3 sequences with distinct hidden states
        for seq in range(3):
            h = np.ones((1, 1, 16), dtype=np.float32) * (seq + 1)
            c = np.ones((1, 1, 16), dtype=np.float32) * (seq + 1) * 10
            buf.set_sequence_hidden(h, c)
            for i in range(3):
                buf.store_transition(np.ones(4) * seq, np.ones(2), 0.0, np.ones(4), False)
        # Sample all 3
        s, a, r, s_, d, h_batch, c_batch = buf.sample_buffer(3)
        # Each h_batch entry should be one of {1, 2, 3}
        for i in range(3):
            val = h_batch[i][0][0][0]
            assert val in [1.0, 2.0, 3.0]

    def test_no_hidden_set_stores_zeros(self):
        buf = SequenceReplayBuffer(
            max_sequence=5, num_observations=4, num_actions=2,
            seq_len=3, hidden_size=16, num_layers=1
        )
        # Don't call set_sequence_hidden — hidden should be zeros
        # Store 2 complete sequences so sample_buffer has a valid pool
        for i in range(6):
            buf.store_transition(np.ones(4), np.ones(2), 0.0, np.ones(4), False)
        s, a, r, s_, d, h_batch, c_batch = buf.sample_buffer(1)
        np.testing.assert_array_equal(h_batch[0], np.zeros((1, 1, 16)))
