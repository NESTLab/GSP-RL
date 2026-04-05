# GSP-RL Configuration Reference

Sources: `gsp_rl/sample_config.yml`, `examples/baselines/cart_pole_config.yml`

All keys are loaded from a YAML file and passed as a dict to `Actor.__init__` and `Hyperparameters.__init__`. Keys not present in `sample_config.yml` are marked with the config file that defines them.

---

## Environment / Training Loop

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `N_GAMES` | int | `500` | Number of training episodes | training loop |

---

## Actor / Network Selection

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `LEARNING_SCHEME` | str | `"DQN"` / `"DDQN"` / `"DDPG"` / `"TD3"` / `"RDDPG"` | Selects which network family to instantiate | `Actor.build_networks()` |
| `INPUT_SIZE` | int | `4` | Observation space dimensionality | `Actor.__init__`, all network constructors |
| `OUTPUT_SIZE` | int | `2` | Action space dimensionality | `Actor.__init__`, all network constructors |
| `MIN_MAX_ACTION` | float | `1.0` | Symmetric action clipping range for continuous policies (`[-v, v]`) | `DDPGActorNetwork`, `TD3ActorNetwork`, `choose_action` |
| `META_PARAM_SIZE` | int | `256` | Output size of `EnvironmentEncoder` (LSTM encoding dimension fed into DDPG actor/critic) | `Actor.__init__`, RDDPG network constructors |

---

## GSP Flags

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `GSP` | bool | `True` / `False` | Enable Global State Prediction; adds `gsp_output_size` dims to main network input | `Actor.__init__` |
| `RECURRENT` | bool | `False` | Use RDDPG-based GSP (sequence replay buffer + LSTM encoder in GSP networks) | `Actor.__init__` as `recurrent_gsp` |
| `ATTENTION` | bool | `False` | Use A-GSP (transformer-based supervised GSP); mutually exclusive with `RECURRENT` | `Actor.__init__` as `attention` |

---

## GSP Network Parameters

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `GSP_INPUT_SIZE` | int | `1` | Observation size for the GSP network | `Actor.__init__` as `gsp_input_size`; `build_gsp_network` |
| `GSP_OUTPUT_SIZE` | int | `1` | Prediction size of the GSP network; concatenated with main observation | `Actor.__init__` as `gsp_output_size`; sets `network_input_size` |
| `GSP_MIN_MAX_ACTION` | float | `1.0` | Action clipping range for GSP actor | `build_DDPG_gsp`, `build_RDDPG_gsp` |
| `GSP_LOOK_BACK` | int | `2` | Look-back window size for GSP observation construction | `Actor.__init__` as `gsp_look_back` |
| `GSP_SEQUENCE_LENGTH` | int | `5` | Sequence length for recurrent / attention GSP replay buffers and `AttentionEncoder.max_length` | `Actor.__init__` as `gsp_sequence_length`; `AttentionSequenceReplayBuffer`, `SequenceReplayBuffer` |

---

## Recurrent (LSTM) Parameters

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `RECURRENT_HIDDEN_SIZE` | int | `256` | LSTM hidden state size in `EnvironmentEncoder` | `EnvironmentEncoder.__init__` as `hidden_size` |
| `RECURRENT_EMBEDDING_SIZE` | int | `256` | Linear embedding layer output size (input to LSTM) in `EnvironmentEncoder` | `EnvironmentEncoder.__init__` as `embedding_size` |
| `RECURRENT_NUM_LAYERS` | int | `5` | Number of stacked LSTM layers in `EnvironmentEncoder` | `EnvironmentEncoder.__init__` as `num_layers` |

---

## Hyperparameters

### Discount and Soft-Update

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `GAMMA` | float | `0.99997` | Discount factor for Bellman targets | all `learn_*` methods |
| `TAU` | float | `0.005` | Polyak averaging rate for soft target-network updates | `update_DDPG_network_parameters`, `update_TD3_network_parameters` |

### Learning Rates

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `ALPHA` | float | `0.001` | Actor learning rate for TD3 | `TD3ActorNetwork` optimizer |
| `BETA` | float | `0.002` | Critic learning rate for TD3 | `TD3CriticNetwork` optimizer |
| `LR` | float | `0.0001` | General learning rate used by DQN, DDQN, DDPG, RDDPG networks and `AttentionEncoder` | all non-TD3 network optimizers |

### Epsilon-Greedy Exploration (DQN / DDQN only)

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `EPSILON` | float | `1.0` | Initial epsilon (probability of random action) | `Actor.choose_action`, `Hyperparameters.__init__` |
| `EPS_MIN` | float | `0.01` | Floor for epsilon decay | `decrement_epsilon()` |
| `EPS_DEC` | float | `0.00005` | Amount subtracted from epsilon after each learn step | `decrement_epsilon()` |

### Replay and Batch

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `BATCH_SIZE` | int | `64` | Number of transitions sampled per main-network learn step | all `learn_*` methods, `sample_memory` |
| `MEM_SIZE` | int | `100000` | Maximum capacity of `ReplayBuffer` | `ReplayBuffer.__init__` |
| `REPLACE_TARGET_COUNTER` | int | `1000` | Steps between hard target-network weight copies (DQN / DDQN) | `replace_target_network()` |

### Continuous Action Exploration

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `NOISE` | float | `0.1` | Standard deviation of Gaussian exploration noise added to actions | `DDPG_choose_action`, `TD3_choose_action` |
| `UPDATE_ACTOR_ITER` | int | `2` | TD3 actor update period: actor updates once every N critic updates | `learn_TD3` |
| `WARMUP` | int | `1000` | Steps of random action before TD3 policy activates | `TD3_choose_action` |

### GSP Learning Schedule

| Key | Type | Default / Example | Description | Used By |
|-----|------|-------------------|-------------|---------|
| `GSP_LEARNING_FREQUENCY` | int | `1000` | Main-network learn steps between each GSP learn call | `Actor.learn()` as `gsp_learning_offset` |
| `GSP_BATCH_SIZE` | int | `16` | Batch size (or number of sequences) used in GSP learn methods | `learn_RDDPG` (loop bound), `learn_attention` (buffer check) |
