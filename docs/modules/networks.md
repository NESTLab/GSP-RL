# networks/ Package

## Overview
7 files, 10 classes. All neural network architectures. Every class extends `nn.Module`.

## Common Patterns
- All networks auto-detect device in constructor: `T.device('cuda:0' if T.cuda.is_available() else 'cpu')`
- All use Adam optimizer with `weight_decay=1e-4`
- All have `save_checkpoint(path, intention=False)` and `load_checkpoint(path, intention=False)`
- The `intention` param appends `"_intention"` to the filename (used for GSP network checkpoints)
- BUG: CPU branch in several load_checkpoint methods calls `load_stat_dict` (typo, missing 'e'). Affects: dqn.py:67, ddqn.py:66, ddpg.py:162, td3.py:69, lstm.py:73, self_attention.py:171

## dqn.py -- DQN
- Class: `DQN(nn.Module)`
- Architecture: `Linear(input_size, 64) -> ReLU -> Linear(64, 128) -> ReLU -> Linear(128, output_size)`
- Constructor args: `id, lr, input_size, output_size, fc1_dims=64, fc2_dims=128, name='DQN'`
- forward(state): `(*, input_size) -> (*, output_size)` -- raw Q-values, no activation on output
- Has its own `self.loss = nn.MSELoss()`

## ddqn.py -- DDQN
- Class: `DDQN(nn.Module)`
- Identical architecture to DQN. Same constructor args (name defaults to 'DDQN').

## ddpg.py -- DDPG Actor + Critic
- Helper: `fanin_init(size, fanin=None)` -- uniform initialization scaled by 1/sqrt(fanin). Used by both actor and critic.
- Class: `DDPGActorNetwork(nn.Module)`
  - Architecture: `Linear(input_size, 400) -> ReLU -> Linear(400, 300) -> ReLU -> Linear(300, output_size) -> Tanh * min_max_action`
  - Constructor args: `id, lr, input_size, output_size, fc1_dims=400, fc2_dims=300, name="DDPG_Actor", min_max_action=1.0`
  - Weight init: fc1, fc2 use fanin_init; mu uses uniform(-3e-3, 3e-3)
  - Name format: `{name}_{id}_DDPG` (e.g., "actor_1_DDPG")
  - forward(x): `(*, input_size) -> (*, output_size)` -- bounded by min_max_action

- Class: `DDPGCriticNetwork(nn.Module)`
  - Architecture: `Linear(input_size, 400) -> ReLU -> Linear(400, 300) -> ReLU -> Linear(300, 1)`
  - IMPORTANT: `input_size` must equal `state_dim + action_dim` because `forward()` concatenates state and action: `T.cat([state, action], dim=-1)`
  - Constructor args: `id, lr, input_size, output_size, fc1_dims=400, fc2_dims=300, name="DDPG_Critic"`
  - forward(state, action): `((*, state_dim), (*, action_dim)) -> (*, 1)` -- Q-value

## td3.py -- TD3 Actor + Critic
- Class: `TD3ActorNetwork(nn.Module)`
  - Same architecture as DDPGActorNetwork but fc1_dims and fc2_dims are required args (no defaults)
  - Uses `alpha` (not `lr`) as learning rate param name
  - Name format: `{name}_{id}_TD3`
  - No fanin_init -- uses default PyTorch initialization
  - forward(state): same signature as DDPGActorNetwork

- Class: `TD3CriticNetwork(nn.Module)`
  - Same architecture as DDPGCriticNetwork but fc1_dims and fc2_dims required
  - Uses `beta` (not `lr`) as learning rate param name
  - forward(state, action): concatenates with `dim=1` (vs DDPG's `dim=-1`, equivalent for 2D)

## lstm.py -- EnvironmentEncoder
- Class: `EnvironmentEncoder(nn.Module)`
  - Architecture: `Linear(input_size, embedding_size) -> LSTM(embedding_size, hidden_size, num_layers, batch_first=True) -> Linear(hidden_size, output_size)`
  - Constructor args: `input_size, output_size, hidden_size, embedding_size, batch_size, num_layers, lr`
  - forward(observation): reshapes embed to `(batch, 1, embedding_size)` before LSTM. Output shape: `(batch, 1, output_size)`
  - Name: "Enviroment_Encoder" (note: typo in source, missing 'n')
  - No optimizer defined (commented out) -- optimizer is on the RDDPG wrapper

## rddpg.py -- Recurrent DDPG Wrappers
- Class: `RDDPGActorNetwork(nn.Module)`
  - Composition: `self.ee = EnvironmentEncoder`, `self.actor = DDPGActorNetwork`
  - forward(x): `ee(x) -> actor(encoding)` -- encoding replaces raw state as DDPG input
  - Has its own Adam optimizer over all parameters (EE + actor)
  - save_checkpoint appends '_recurrent' to path

- Class: `RDDPGCriticNetwork(nn.Module)`
  - Composition: `self.ee = EnvironmentEncoder`, `self.critic = DDPGCriticNetwork`
  - forward(state, action): `ee(state) -> critic(encoding, action)`
  - NOTE: In save_checkpoint, EE is NOT saved (comment says "saved in actor network"). In load_checkpoint, EE IS loaded.

## self_attention.py -- Attention-Based Encoder
- Class: `SelfAttention(nn.Module)`
  - Multi-head self-attention. embed_size must be divisible by heads.
  - head_dim = embed_size // heads
  - Projections: values, keys, query -- all Linear(head_dim, head_dim, bias=False)
  - Uses Einstein summation for attention: `einsum("nqhd, nkhd->nhqk")`
  - forward(values, keys, query, mask=None): all inputs shape `(N, seq_len, embed_size)`

- Class: `TransformerBlock(nn.Module)`
  - SelfAttention + LayerNorm + FeedForward + LayerNorm + Dropout
  - forward(value, key, query, mask=None): skip connections on both attention and feedforward

- Class: `AttentionEncoder(nn.Module)`
  - word_embedding: `Linear(input_size, hidden_size) -> ReLU -> Linear(hidden_size, 1)` -- reduces each obs to scalar
  - position_embedding: `nn.Embedding(max_length, embed_size)`
  - Single TransformerBlock layer
  - fc_out: `Linear(embed_size * max_length, output_size)` -- flattened transformer output
  - forward(x, mask=None): input `(N, seq_len, obs_size)` -> output `(N, output_size)`, bounded by tanh * min_max_action
  - Hardcoded lr=0.0001 in optimizer (not from config)
  - Name: "Attention_Encoder"

## __init__.py Exports
```python
from .dqn import DQN
from .ddqn import DDQN
from .ddpg import DDPGActorNetwork, DDPGCriticNetwork
from .rddpg import RDDPGActorNetwork, RDDPGCriticNetwork
from .lstm import EnvironmentEncoder
from .td3 import TD3ActorNetwork, TD3CriticNetwork
from .self_attention import AttentionEncoder
```
