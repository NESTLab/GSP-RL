"""Tests for SequenceReplayBuffer (gsp_rl/src/buffers/sequential.py)."""
import numpy as np
import pytest

from gsp_rl.src.buffers.sequential import SequenceReplayBuffer


NUM_OBS = 4
NUM_ACTIONS = 2
MAX_SEQUENCE = 10
SEQ_LEN = 5


def make_buffer(max_sequence=MAX_SEQUENCE, seq_len=SEQ_LEN):
    return SequenceReplayBuffer(max_sequence, NUM_OBS, NUM_ACTIONS, seq_len)


def sample_transition(obs_size=NUM_OBS, num_actions=NUM_ACTIONS):
    state = np.random.rand(obs_size)
    state_ = np.random.rand(obs_size)
    action = np.random.rand(num_actions)
    reward = float(np.random.rand())
    done = bool(np.random.randint(0, 2))
    return state, action, reward, state_, done


def fill_n_sequences(buf, n_sequences):
    """Fill exactly n complete sequences into the buffer."""
    for _ in range(n_sequences):
        for _ in range(buf.seq_len):
            s, a, r, s_, d = sample_transition()
            buf.store_transition(s, a, r, s_, d)


class TestSequenceReplayBufferInitialState:
    def test_mem_size_equals_max_sequence_times_seq_len(self):
        buf = make_buffer()
        assert buf.mem_size == MAX_SEQUENCE * SEQ_LEN

    def test_mem_ctr_starts_at_zero(self):
        buf = make_buffer()
        assert buf.mem_ctr == 0

    def test_seq_mem_cntr_starts_at_zero(self):
        buf = make_buffer()
        assert buf.seq_mem_cntr == 0

    def test_state_memory_shape(self):
        buf = make_buffer()
        assert buf.state_memory.shape == (MAX_SEQUENCE * SEQ_LEN, NUM_OBS)

    def test_action_memory_shape(self):
        buf = make_buffer()
        assert buf.action_memory.shape == (MAX_SEQUENCE * SEQ_LEN, NUM_ACTIONS)

    def test_reward_memory_shape(self):
        buf = make_buffer()
        assert buf.reward_memory.shape == (MAX_SEQUENCE * SEQ_LEN,)

    def test_terminal_memory_shape(self):
        buf = make_buffer()
        assert buf.terminal_memory.shape == (MAX_SEQUENCE * SEQ_LEN,)

    def test_seq_state_memory_shape(self):
        buf = make_buffer()
        assert buf.seq_state_memory.shape == (SEQ_LEN, NUM_OBS)

    def test_seq_action_memory_shape(self):
        buf = make_buffer()
        assert buf.seq_action_memory.shape == (SEQ_LEN, NUM_ACTIONS)


class TestSequenceReplayBufferStaging:
    def test_staging_does_not_flush_before_seq_len(self):
        buf = make_buffer()
        # Store seq_len - 1 transitions — should not flush
        for _ in range(SEQ_LEN - 1):
            s, a, r, s_, d = sample_transition()
            buf.store_transition(s, a, r, s_, d)
        assert buf.mem_ctr == 0

    def test_staging_seq_mem_cntr_increments(self):
        buf = make_buffer()
        for i in range(SEQ_LEN - 1):
            s, a, r, s_, d = sample_transition()
            buf.store_transition(s, a, r, s_, d)
        assert buf.seq_mem_cntr == SEQ_LEN - 1

    def test_staging_flushes_at_seq_len(self):
        buf = make_buffer()
        for _ in range(SEQ_LEN):
            s, a, r, s_, d = sample_transition()
            buf.store_transition(s, a, r, s_, d)
        assert buf.mem_ctr == SEQ_LEN

    def test_staging_resets_seq_mem_cntr_after_flush(self):
        buf = make_buffer()
        for _ in range(SEQ_LEN):
            s, a, r, s_, d = sample_transition()
            buf.store_transition(s, a, r, s_, d)
        assert buf.seq_mem_cntr == 0

    def test_partial_sequence_after_flush_resumes_staging(self):
        buf = make_buffer()
        # Complete one sequence
        for _ in range(SEQ_LEN):
            s, a, r, s_, d = sample_transition()
            buf.store_transition(s, a, r, s_, d)
        # Store 2 more (partial second sequence)
        for _ in range(2):
            s, a, r, s_, d = sample_transition()
            buf.store_transition(s, a, r, s_, d)
        assert buf.seq_mem_cntr == 2
        assert buf.mem_ctr == SEQ_LEN  # main buffer unchanged

    def test_two_complete_sequences_flush_correctly(self):
        buf = make_buffer()
        fill_n_sequences(buf, 2)
        assert buf.mem_ctr == 2 * SEQ_LEN


class TestSequenceReplayBufferFlushedData:
    def test_flushed_states_match_stored_values(self):
        buf = make_buffer(seq_len=3)
        states = [np.array([float(i), float(i), float(i), float(i)]) for i in range(3)]
        actions = [np.zeros(NUM_ACTIONS) for _ in range(3)]
        for i in range(3):
            buf.store_transition(states[i], actions[i], 0.0, states[i], False)
        # After flush, states should be in main buffer at indices 0, 1, 2
        for i in range(3):
            np.testing.assert_array_almost_equal(buf.state_memory[i], states[i])

    def test_flushed_actions_match_stored_values(self):
        buf = make_buffer(seq_len=3)
        state = np.zeros(NUM_OBS)
        actions = [np.array([float(i), float(i)]) for i in range(3)]
        for i in range(3):
            buf.store_transition(state, actions[i], 0.0, state, False)
        for i in range(3):
            np.testing.assert_array_almost_equal(buf.action_memory[i], actions[i])

    def test_flushed_rewards_match_stored_values(self):
        buf = make_buffer(seq_len=3)
        state = np.zeros(NUM_OBS)
        action = np.zeros(NUM_ACTIONS)
        rewards = [1.0, 2.0, 3.0]
        for r in rewards:
            buf.store_transition(state, action, r, state, False)
        for i, r in enumerate(rewards):
            assert buf.reward_memory[i] == pytest.approx(r)

    def test_flushed_dones_match_stored_values(self):
        buf = make_buffer(seq_len=3)
        state = np.zeros(NUM_OBS)
        action = np.zeros(NUM_ACTIONS)
        dones = [True, False, True]
        for d in dones:
            buf.store_transition(state, action, 0.0, state, d)
        for i, d in enumerate(dones):
            assert bool(buf.terminal_memory[i]) == d


class TestSequenceReplayBufferSample:
    def _fill_enough(self, buf, n_sequences=5):
        fill_n_sequences(buf, n_sequences)

    def test_sample_returns_five_arrays(self):
        buf = make_buffer()
        self._fill_enough(buf)
        result = buf.sample_buffer(batch_size=2)
        assert len(result) == 5

    def test_sample_states_shape(self):
        buf = make_buffer()
        self._fill_enough(buf)
        s, *_ = buf.sample_buffer(batch_size=2)
        assert s.shape == (2, SEQ_LEN, NUM_OBS)

    def test_sample_actions_shape(self):
        buf = make_buffer()
        self._fill_enough(buf)
        _, a, *_ = buf.sample_buffer(batch_size=2)
        assert a.shape == (2, SEQ_LEN, NUM_ACTIONS)

    def test_sample_rewards_shape(self):
        buf = make_buffer()
        self._fill_enough(buf)
        _, _, r, *_ = buf.sample_buffer(batch_size=2)
        assert r.shape == (2, SEQ_LEN)

    def test_sample_next_states_shape(self):
        buf = make_buffer()
        self._fill_enough(buf)
        _, _, _, s_, _ = buf.sample_buffer(batch_size=2)
        assert s_.shape == (2, SEQ_LEN, NUM_OBS)

    def test_sample_dones_shape(self):
        buf = make_buffer()
        self._fill_enough(buf)
        *_, d = buf.sample_buffer(batch_size=2)
        assert d.shape == (2, SEQ_LEN)

    def test_sample_larger_batch_size(self):
        buf = make_buffer()
        self._fill_enough(buf, n_sequences=8)
        s, *_ = buf.sample_buffer(batch_size=4)
        assert s.shape == (4, SEQ_LEN, NUM_OBS)
