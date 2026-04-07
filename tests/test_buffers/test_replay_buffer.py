"""Tests for ReplayBuffer (gsp_rl/src/buffers/replay.py)."""
import numpy as np
import pytest

from gsp_rl.src.buffers.replay import ReplayBuffer


NUM_OBS = 4
NUM_ACTIONS = 2
MAX_SIZE = 10


def make_discrete_buffer(max_size=MAX_SIZE):
    return ReplayBuffer(max_size, NUM_OBS, NUM_ACTIONS, action_type="Discrete")


def make_continuous_buffer(max_size=MAX_SIZE):
    return ReplayBuffer(max_size, NUM_OBS, NUM_ACTIONS, action_type="Continuous")


def sample_transition(obs_size=NUM_OBS):
    state = np.random.rand(obs_size).astype(np.float32)
    state_ = np.random.rand(obs_size).astype(np.float32)
    reward = float(np.random.rand())
    done = bool(np.random.randint(0, 2))
    return state, reward, state_, done


class TestReplayBufferInitialState:
    def test_mem_ctr_starts_at_zero(self):
        buf = make_discrete_buffer()
        assert buf.mem_ctr == 0

    def test_state_memory_shape(self):
        buf = make_discrete_buffer()
        assert buf.state_memory.shape == (MAX_SIZE, NUM_OBS)

    def test_new_state_memory_shape(self):
        buf = make_discrete_buffer()
        assert buf.new_state_memory.shape == (MAX_SIZE, NUM_OBS)

    def test_reward_memory_shape(self):
        buf = make_discrete_buffer()
        assert buf.reward_memory.shape == (MAX_SIZE,)

    def test_terminal_memory_shape(self):
        buf = make_discrete_buffer()
        assert buf.terminal_memory.shape == (MAX_SIZE,)

    def test_discrete_action_memory_shape(self):
        buf = make_discrete_buffer()
        assert buf.action_memory.shape == (MAX_SIZE,)

    def test_discrete_action_memory_dtype_is_int(self):
        buf = make_discrete_buffer()
        assert np.issubdtype(buf.action_memory.dtype, np.integer)

    def test_continuous_action_memory_shape(self):
        buf = make_continuous_buffer()
        assert buf.action_memory.shape == (MAX_SIZE, NUM_ACTIONS)

    def test_continuous_action_memory_dtype_is_float(self):
        buf = make_continuous_buffer()
        assert np.issubdtype(buf.action_memory.dtype, np.floating)

    def test_unknown_action_type_raises_exception(self):
        with pytest.raises(Exception):
            ReplayBuffer(MAX_SIZE, NUM_OBS, NUM_ACTIONS, action_type="Unknown")


class TestReplayBufferStoreAndRetrieve:
    def test_store_transition_increments_mem_ctr(self):
        buf = make_discrete_buffer()
        state, reward, state_, done = sample_transition()
        buf.store_transition(state, 1, reward, state_, done)
        assert buf.mem_ctr == 1

    def test_stored_state_matches(self):
        buf = make_discrete_buffer()
        state, reward, state_, done = sample_transition()
        buf.store_transition(state, 1, reward, state_, done)
        np.testing.assert_array_almost_equal(buf.state_memory[0], state)

    def test_stored_next_state_matches(self):
        buf = make_discrete_buffer()
        state, reward, state_, done = sample_transition()
        buf.store_transition(state, 1, reward, state_, done)
        np.testing.assert_array_almost_equal(buf.new_state_memory[0], state_)

    def test_stored_reward_matches(self):
        buf = make_discrete_buffer()
        state, reward, state_, done = sample_transition()
        buf.store_transition(state, 1, reward, state_, done)
        assert buf.reward_memory[0] == pytest.approx(reward)

    def test_stored_done_matches(self):
        buf = make_discrete_buffer()
        state, reward, state_, done = sample_transition()
        buf.store_transition(state, 1, reward, state_, done)
        assert bool(buf.terminal_memory[0]) == done

    def test_stored_discrete_action_matches(self):
        buf = make_discrete_buffer()
        state, reward, state_, done = sample_transition()
        buf.store_transition(state, 3, reward, state_, done)
        assert buf.action_memory[0] == 3

    def test_stored_continuous_action_matches(self):
        buf = make_continuous_buffer()
        state, reward, state_, done = sample_transition()
        action = np.array([0.5, -0.3], dtype=np.float32)
        buf.store_transition(state, action, reward, state_, done)
        np.testing.assert_array_almost_equal(buf.action_memory[0], action)

    def test_second_transition_stored_at_index_one(self):
        buf = make_discrete_buffer()
        for _ in range(2):
            state, reward, state_, done = sample_transition()
            buf.store_transition(state, 0, reward, state_, done)
        assert buf.mem_ctr == 2

    def test_store_multiple_transitions_increments_correctly(self):
        buf = make_discrete_buffer()
        n = 5
        for _ in range(n):
            state, reward, state_, done = sample_transition()
            buf.store_transition(state, 0, reward, state_, done)
        assert buf.mem_ctr == n


class TestReplayBufferCircularOverwrite:
    def test_mem_ctr_exceeds_mem_size_after_overflow(self):
        buf = make_discrete_buffer(max_size=5)
        for _ in range(7):
            state, reward, state_, done = sample_transition()
            buf.store_transition(state, 0, reward, state_, done)
        assert buf.mem_ctr == 7

    def test_circular_index_wraps_to_zero(self):
        buf = make_discrete_buffer(max_size=5)
        # Fill buffer completely
        for i in range(5):
            state = np.full(NUM_OBS, float(i), dtype=np.float32)
            buf.store_transition(state, i, float(i), state, False)
        # One more — should overwrite index 0
        overwrite_state = np.full(NUM_OBS, 99.0, dtype=np.float32)
        buf.store_transition(overwrite_state, 0, 99.0, overwrite_state, True)
        np.testing.assert_array_almost_equal(buf.state_memory[0], overwrite_state)

    def test_previous_entries_intact_after_partial_overwrite(self):
        buf = make_discrete_buffer(max_size=5)
        states = [np.full(NUM_OBS, float(i), dtype=np.float32) for i in range(5)]
        for i, s in enumerate(states):
            buf.store_transition(s, i, float(i), s, False)
        # Overwrite index 0 only
        new_state = np.full(NUM_OBS, 99.0, dtype=np.float32)
        buf.store_transition(new_state, 0, 99.0, new_state, True)
        # Index 1 should still have value from the second original transition
        np.testing.assert_array_almost_equal(buf.state_memory[1], states[1])


class TestReplayBufferSample:
    def _fill_buffer(self, buf, n):
        for i in range(n):
            state = np.random.rand(NUM_OBS).astype(np.float32)
            buf.store_transition(state, 0, float(i), state, False)

    def test_sample_returns_five_arrays(self):
        buf = make_discrete_buffer()
        self._fill_buffer(buf, MAX_SIZE)
        result = buf.sample_buffer(batch_size=4)
        assert len(result) == 5

    def test_sample_states_shape(self):
        buf = make_discrete_buffer()
        self._fill_buffer(buf, MAX_SIZE)
        states, *_ = buf.sample_buffer(batch_size=4)
        assert states.shape == (4, NUM_OBS)

    def test_sample_discrete_actions_shape(self):
        buf = make_discrete_buffer()
        self._fill_buffer(buf, MAX_SIZE)
        _, actions, *_ = buf.sample_buffer(batch_size=4)
        assert actions.shape == (4,)

    def test_sample_continuous_actions_shape(self):
        buf = make_continuous_buffer()
        self._fill_buffer(buf, MAX_SIZE)
        _, actions, *_ = buf.sample_buffer(batch_size=4)
        assert actions.shape == (4, NUM_ACTIONS)

    def test_sample_rewards_shape(self):
        buf = make_discrete_buffer()
        self._fill_buffer(buf, MAX_SIZE)
        _, _, rewards, *_ = buf.sample_buffer(batch_size=4)
        assert rewards.shape == (4,)

    def test_sample_next_states_shape(self):
        buf = make_discrete_buffer()
        self._fill_buffer(buf, MAX_SIZE)
        _, _, _, next_states, _ = buf.sample_buffer(batch_size=4)
        assert next_states.shape == (4, NUM_OBS)

    def test_sample_dones_shape(self):
        buf = make_discrete_buffer()
        self._fill_buffer(buf, MAX_SIZE)
        *_, dones = buf.sample_buffer(batch_size=4)
        assert dones.shape == (4,)

    def test_sample_without_replacement_unique_indices(self):
        """Verify sampled batch has no duplicate entries (sample without replace)."""
        buf = make_discrete_buffer(max_size=20)
        # Store unique states so we can detect duplicates
        for i in range(20):
            state = np.full(NUM_OBS, float(i), dtype=np.float32)
            buf.store_transition(state, 0, float(i), state, False)
        states, _, rewards, _, _ = buf.sample_buffer(batch_size=10)
        # Rewards were set to the index value, so they should be unique
        assert len(np.unique(rewards)) == 10

    def test_sample_batch_size_equal_to_buffer_size(self):
        buf = make_discrete_buffer(max_size=MAX_SIZE)
        self._fill_buffer(buf, MAX_SIZE)
        states, *_ = buf.sample_buffer(batch_size=MAX_SIZE)
        assert states.shape == (MAX_SIZE, NUM_OBS)
