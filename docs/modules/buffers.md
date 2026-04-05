# buffers/ Package

## Overview
3 files, 3 classes. Experience replay storage for training.

## replay.py -- ReplayBuffer
- Standard SARSD circular buffer
- Constructor: `max_size, num_observations, num_actions, action_type='Discrete'|'Continuous'`
- Storage: numpy arrays. Discrete actions stored as int (1D), Continuous as float32 (2D: mem_size x num_actions)
- `store_transition(state, action, reward, state_, done)`: circular write at `mem_ctr % mem_size`
- `sample_buffer(batch_size)` -> `(states, actions, rewards, next_states, dones)` as numpy arrays. Random sampling without replacement.
- `mem_ctr`: total transitions stored (not bounded by mem_size)

## sequential.py -- SequenceReplayBuffer
- Two-stage buffer for temporal sequence data (used by RDDPG-GSP)
- Constructor: `max_sequence, num_observations, num_actions, seq_len`
- Total capacity: `mem_size = max_sequence * seq_len`
- Stage 1: `seq_*_memory` arrays of shape `(seq_len, ...)` -- accumulates one sequence
- Stage 2: when `seq_mem_cntr == seq_len`, copies entire sequence to main buffer, resets seq counter
- `store_transition(s, a, r, s_, d)`: stores to staging buffer; flushes to main when full
- `sample_buffer(batch_size)` -> `(s, a, r, s_, d)` with shapes `(batch_size, seq_len, *)`. Samples sequence start indices (multiples of seq_len).
- `get_current_sequence()` -> latest complete sequence from main buffer

## attention_sequential.py -- AttentionSequenceReplayBuffer
- Stores observation sequences with scalar labels (supervised, not SARSD)
- Constructor: `num_observations, seq_len`. Hardcoded `mem_size=10000`.
- Storage: `state_memory (mem_size, num_observations)` + `label_memory (mem_size,)`
- `store_transition(s, y)`: s is one observation, y is a label. Stages in seq buffer; flushes seq_len observations + 1 label to main.
- `sample_buffer(batch_size)` -> `(observations, labels)` with shapes `(batch_size, seq_len, num_observations)` and `(batch_size,)`
- Label is stored only at the sequence start index in main buffer

## __init__.py Exports
```python
from .replay import ReplayBuffer
from .sequential import SequenceReplayBuffer
from .attention_sequential import AttentionSequenceReplayBuffer
```
