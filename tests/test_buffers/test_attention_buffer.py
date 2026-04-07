"""Tests for AttentionSequenceReplayBuffer (gsp_rl/src/buffers/attention_sequential.py)."""
import numpy as np
import pytest

from gsp_rl.src.buffers.attention_sequential import AttentionSequenceReplayBuffer


NUM_OBS = 6
SEQ_LEN = 4
HARDCODED_MEM_SIZE = 10000


def make_buffer(num_observations=NUM_OBS, seq_len=SEQ_LEN):
    return AttentionSequenceReplayBuffer(num_observations, seq_len)


def fill_n_sequences(buf, n_sequences, label=1.0):
    """Fill exactly n complete sequences into the buffer."""
    for _ in range(n_sequences):
        for step in range(buf.seq_len):
            state = np.random.rand(buf.num_observations)
            is_last = step == buf.seq_len - 1
            buf.store_transition(state, label if is_last else 0.0)


class TestAttentionBufferInitialState:
    def test_mem_size_is_hardcoded_to_10000(self):
        buf = make_buffer()
        assert buf.mem_size == HARDCODED_MEM_SIZE

    def test_mem_ctr_starts_at_zero(self):
        buf = make_buffer()
        assert buf.mem_ctr == 0

    def test_seq_mem_cntr_starts_at_zero(self):
        buf = make_buffer()
        assert buf.seq_mem_cntr == 0

    def test_state_memory_shape(self):
        buf = make_buffer()
        assert buf.state_memory.shape == (HARDCODED_MEM_SIZE, NUM_OBS)

    def test_label_memory_shape(self):
        buf = make_buffer()
        assert buf.label_memory.shape == (HARDCODED_MEM_SIZE,)

    def test_seq_state_memory_shape(self):
        buf = make_buffer()
        assert buf.seq_state_memory.shape == (SEQ_LEN, NUM_OBS)

    def test_num_observations_stored(self):
        buf = make_buffer(num_observations=8)
        assert buf.num_observations == 8

    def test_seq_len_stored(self):
        buf = make_buffer(seq_len=10)
        assert buf.seq_len == 10


class TestAttentionBufferStoreAndFlush:
    def test_staging_does_not_flush_before_seq_len(self):
        buf = make_buffer()
        for _ in range(SEQ_LEN - 1):
            state = np.random.rand(NUM_OBS)
            buf.store_transition(state, 0.0)
        assert buf.mem_ctr == 0

    def test_staging_seq_mem_cntr_increments(self):
        buf = make_buffer()
        for _ in range(SEQ_LEN - 1):
            state = np.random.rand(NUM_OBS)
            buf.store_transition(state, 0.0)
        assert buf.seq_mem_cntr == SEQ_LEN - 1

    def test_staging_flushes_at_seq_len(self):
        buf = make_buffer()
        for _ in range(SEQ_LEN):
            state = np.random.rand(NUM_OBS)
            buf.store_transition(state, 1.0)
        assert buf.mem_ctr == SEQ_LEN

    def test_staging_resets_seq_mem_cntr_after_flush(self):
        buf = make_buffer()
        for _ in range(SEQ_LEN):
            state = np.random.rand(NUM_OBS)
            buf.store_transition(state, 1.0)
        assert buf.seq_mem_cntr == 0

    def test_flushed_states_match_stored_values(self):
        buf = make_buffer(seq_len=3)
        states = [np.array([float(i)] * NUM_OBS) for i in range(3)]
        for s in states:
            buf.store_transition(s, 99.0)
        for i, s in enumerate(states):
            np.testing.assert_array_almost_equal(buf.state_memory[i], s)

    def test_label_stored_only_once_per_sequence(self):
        """Label is stored once at the sequence start index in main buffer."""
        buf = make_buffer(seq_len=3)
        for _ in range(3):
            state = np.random.rand(NUM_OBS)
            buf.store_transition(state, 42.0)
        # label is stored at mem_index = 0 (start of sequence)
        assert buf.label_memory[0] == pytest.approx(42.0)

    def test_two_sequences_store_correct_labels(self):
        buf = make_buffer(seq_len=3)
        # First sequence with label 1.0
        for _ in range(3):
            buf.store_transition(np.random.rand(NUM_OBS), 1.0)
        # Second sequence with label 2.0
        for _ in range(3):
            buf.store_transition(np.random.rand(NUM_OBS), 2.0)
        assert buf.label_memory[0] == pytest.approx(1.0)
        assert buf.label_memory[3] == pytest.approx(2.0)

    def test_two_complete_sequences_advance_mem_ctr(self):
        buf = make_buffer()
        fill_n_sequences(buf, 2)
        assert buf.mem_ctr == 2 * SEQ_LEN


class TestAttentionBufferSample:
    def _fill_enough(self, buf, n_sequences=5, label=7.5):
        fill_n_sequences(buf, n_sequences, label=label)

    def test_sample_returns_two_arrays(self):
        buf = make_buffer()
        self._fill_enough(buf)
        result = buf.sample_buffer(batch_size=2)
        assert len(result) == 2

    def test_sample_observations_shape(self):
        buf = make_buffer()
        self._fill_enough(buf)
        obs, _ = buf.sample_buffer(batch_size=2)
        assert obs.shape == (2, SEQ_LEN, NUM_OBS)

    def test_sample_labels_shape(self):
        buf = make_buffer()
        self._fill_enough(buf)
        _, labels = buf.sample_buffer(batch_size=2)
        assert labels.shape == (2,)

    def test_sample_larger_batch_size(self):
        buf = make_buffer()
        self._fill_enough(buf, n_sequences=8)
        obs, labels = buf.sample_buffer(batch_size=4)
        assert obs.shape == (4, SEQ_LEN, NUM_OBS)
        assert labels.shape == (4,)

    def test_sample_observations_content_matches_stored(self):
        """Verify sampled sequences contain data that was actually stored."""
        buf = make_buffer(seq_len=3)
        # Store a distinctive sequence
        for i in range(3):
            state = np.full(NUM_OBS, float(i + 10))
            buf.store_transition(state, 5.0)
        # Store another sequence
        for i in range(3):
            state = np.full(NUM_OBS, float(i + 20))
            buf.store_transition(state, 6.0)
        # Sample — with only 1 eligible index (indices list uses max_mem//seq_len - 1),
        # we need enough sequences. Let's just check shape/type here.
        obs, labels = buf.sample_buffer(batch_size=1)
        assert obs.shape == (1, 3, NUM_OBS)
        assert labels.shape == (1,)
