# actors/ Package

## Overview
2 files, 3 classes. The agent layer. `Actor` is the main public API.

## learning_aids.py

### Hyperparameters (base class)
- Constructor takes a `config: dict` (YAML-loaded)
- Stores: gamma, tau, alpha, beta, lr, epsilon, eps_min, eps_dec, gsp_learning_offset, gsp_batch_size, batch_size, mem_size, replace_target_ctr, noise, update_actor_iter, warmup, time_step=0
- `config['GSP_LEARNING_FREQUENCY']` maps to `self.gsp_learning_offset`
- `config['REPLACE_TARGET_COUNTER']` maps to `self.replace_target_ctr`

### NetworkAids(Hyperparameters)
Network factory methods -- all return plain dicts:
- `make_DQN_networks(nn_args)` -> `{'q_eval': DQN, 'q_next': DQN}`
- `make_DDQN_networks(nn_args)` -> `{'q_eval': DDQN, 'q_next': DDQN}`
- `make_DDPG_networks(actor_nn_args, critic_nn_args)` -> `{'actor', 'target_actor', 'critic', 'target_critic'}`
- `make_TD3_networks(actor_nn_args, critic_nn_args)` -> `{'actor', 'target_actor', 'critic_1', 'target_critic_1', 'critic_2', 'target_critic_2'}`
- `make_RDDPG_networks(lstm_nn_args, actor_nn_args, critic_nn_args)` -> same keys as DDPG, actor/critic share one EnvironmentEncoder, targets get separate ones
- `make_Attention_Encoder(nn_args)` -> `{'attention': AttentionEncoder}`

Soft update methods:
- `update_DDPG_network_parameters(tau, networks)`: Polyak averaging on actor + critic. Iterates parameters directly.
- `update_TD3_network_parameters(tau, networks)`: Same but for TD3's twin critics. Uses named_parameters + load_state_dict pattern.

Action selection methods:
- `DQN_DDQN_choose_action(obs, networks)` -> int: argmax of q_eval forward pass
- `DDPG_choose_action(obs, networks)` -> Tensor: actor forward, unsqueezed. Handles RDDPG case (np.array wrapping).
- `TD3_choose_action(obs, networks, n_actions)` -> numpy: random during warmup, actor + noise after. Returns unsqueezed.
- `Attention_choose_action(obs, networks)` -> numpy: attention forward, detached.

Learn methods: see `docs/algorithms.md` for detailed pseudocode.

Memory methods:
- `sample_memory(networks)` -> (states, actions, rewards, states_, dones) as Tensors on correct device
- `sample_attention_memory(networks)` -> (observations, labels) as Tensors
- `store_transition(s, a, r, s_, d, networks)`: delegates to `networks['replay'].store_transition`
- `store_attention_transition(s, y, networks)`: delegates to `networks['replay'].store_transition(s, y)`
- `decrement_epsilon()`: `epsilon = max(epsilon - eps_dec, eps_min)`

Module-level: `Loss = nn.MSELoss()` -- singleton used by all learn methods.

## actor.py

### Actor(NetworkAids)
Constructor args: `id, config, network, input_size, output_size, min_max_action, meta_param_size, gsp=False, recurrent_gsp=False, attention=False, recurrent_hidden_size=256, recurrent_embedding_size=256, recurrent_num_layers=5, gsp_input_size=6, gsp_output_size=1, gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5`

Key init logic:
1. Calls `super().__init__(config)` -> loads hyperparams
2. Computes `network_input_size = input_size + gsp_output_size` if GSP enabled
3. If attention_gsp: creates `attention_observation` sliding window (list of zero-lists)
4. Calls `build_networks(network)` for main action network
5. BUG: `if gsp is not None` (line 89) is always True. Always calls `build_gsp_network('DDPG')`.

`build_networks(learning_scheme: str)`:
- Dispatches on string: 'None', 'DQN', 'DDQN', 'DDPG', 'RDDPG', 'TD3'
- Constructs nn_args dicts and delegates to `build_DQN/DDQN/DDPG/RDDPG/TD3`
- Adds `'replay'`, `'learning_scheme'`, `'learn_step_counter'` to the returned dict
- BUG: RDDPG path calls `self.build_RDDPG()` with 0 args but method needs 3
- Note: DDPG critic input_size = network_input_size + output_size (for state-action concat)

`build_gsp_network(learning_scheme: str)`:
- Resets `self.gsp_networks = None` at top
- If attention_gsp: builds AttentionEncoder with hardcoded params (embed_size=256, heads=8, etc.)
- Else if 'DDPG': builds DDPG-GSP or RDDPG-GSP depending on recurrent_gsp flag
- Else if 'TD3': builds TD3-GSP or RTD3-GSP

`choose_action(observation, networks, test=False)`:
- Dispatches on `networks['learning_scheme']`
- DQN/DDQN: epsilon-greedy (random if not test and rand < epsilon)
- DDPG/RDDPG: actor forward + Gaussian noise (unless test), clamped
- TD3: delegates to TD3_choose_action
- attention: maintains sliding window, converts to tensor, delegates

`learn()`:
- Returns early if replay buffer has fewer than batch_size samples
- If GSP: triggers learn_gsp() every gsp_learning_offset learn steps
- Dispatches to learn_DQN/DDQN/DDPG/TD3 based on learning_scheme
- DQN/DDQN also call replace_target_network() first
- DDPG/TD3 call update_network_parameters() first

`save_model(path)` / `load_model(path)`:
- Saves/loads all networks in both self.networks and self.gsp_networks
- Dispatches based on learning_scheme strings
