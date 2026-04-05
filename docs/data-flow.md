# GSP-RL Data Flow and Tensor Shapes

This document traces tensor shapes through each algorithm path from environment observation to loss computation.

---

## DQN / DDQN Path

Source: `learning_aids.py:163-217`

| Stage | Shape | Notes |
|-------|-------|-------|
| `observation` (env output) | `(input_size,)` | numpy array |
| `state` (choose_action) | `(input_size,)` | `T.tensor(observation)`, 1-D |
| `action_values` | `(output_size,)` | `q_eval.forward(state)` |
| `states` (after `sample_memory`) | `(batch_size, input_size)` | float32 tensor |
| `actions` | `(batch_size,)` | LongTensor after `.type(T.LongTensor)` |
| `q_eval(states)` | `(batch_size, output_size)` | all Q-values per action |
| `q_pred` | `(batch_size,)` | indexed by `[indices, actions]` |
| `q_next(states_)` (DQN) | `(batch_size, output_size)` | `.max(dim=1)[0]` reduces to `(batch_size,)` |
| `q_eval(states_)` (DDQN) | `(batch_size, output_size)` | used to select `max_actions` |
| `q_next[indices, max_actions]` (DDQN) | `(batch_size,)` | evaluated by q_next at selected actions |
| `q_target` | `(batch_size,)` | `rewards + gamma * q_next` |
| `loss` | scalar | `MSE(q_target, q_pred)` |

---

## DDPG Path

Source: `learning_aids.py:219-245`, `gsp_rl/src/networks/ddpg.py`

| Stage | Shape | Notes |
|-------|-------|-------|
| `states` | `(batch_size, input_size)` | from `sample_memory` |
| `actions` | `(batch_size, output_size)` | float32 tensor |
| `target_actor(states_)` | `(batch_size, output_size)` | tanh * min_max_action |
| critic input | `(batch_size, input_size + output_size)` | state and action concatenated along `dim=-1` inside `DDPGCriticNetwork` |
| `target_critic(states_, target_actions)` | `(batch_size, 1)` | Q-value estimate |
| `rewards` (raw) | `(batch_size,)` | 1-D after `sample_memory` |
| `unsqueeze(rewards, 1)` | `(batch_size, 1)` | broadcast to match critic output |
| `target` | `(batch_size, 1)` | `unsqueeze(rewards,1) + gamma * q_value_` |
| `q_value = critic(states, actions)` | `(batch_size, 1)` | current critic prediction |
| `value_loss` | scalar | `MSE(q_value, target)` |
| `new_policy_actions = actor(states)` | `(batch_size, output_size)` | |
| `actor_loss = -critic(states, new_actions).mean()` | scalar | gradient ascent on critic |

---

## TD3 Path

Source: `learning_aids.py:294-339`

| Stage | Shape | Notes |
|-------|-------|-------|
| `states` | `(batch_size, input_size)` | from `sample_memory` |
| `target_actor(states_)` | `(batch_size, output_size)` | before noise |
| target actions (with noise) | `(batch_size, output_size)` | `+ clamp(Normal(0,0.2), -0.5, 0.5)`, then clamped to `[-min_max_action, min_max_action]` |
| `q1_ = target_critic_1(states_, target_actions)` | `(batch_size, 1)` -> `(batch_size,)` | `.view(-1)` flattens |
| `q2_ = target_critic_2(states_, target_actions)` | `(batch_size, 1)` -> `(batch_size,)` | `.view(-1)` flattens |
| `critic_value_ = min(q1_, q2_)` | `(batch_size,)` | element-wise minimum |
| `target` | `(batch_size,)` | `rewards + gamma * critic_value_` |
| `q1 = critic_1(states, actions).squeeze()` | `(batch_size,)` | squeezed from `(batch_size, 1)` |
| `q2 = critic_2(states, actions).squeeze()` | `(batch_size,)` | squeezed from `(batch_size, 1)` |
| `critic_loss` | scalar | `MSE(target, q1) + MSE(target, q2)` |
| `actor_q1_loss = critic_1(states, actor(states))` | `(batch_size, 1)` | only computed every `update_actor_iter` steps |
| `actor_loss = -mean(actor_q1_loss)` | scalar | |

---

## RDDPG Path

Source: `learning_aids.py:247-292`, `gsp_rl/src/networks/lstm.py`, `gsp_rl/src/networks/rddpg.py`

Outer loop iterates over each sequence in the batch individually.

| Stage | Shape | Notes |
|-------|-------|-------|
| `s` (from `SequenceReplayBuffer`) | `(batch_size, seq_len, input_size)` | |
| `states = s[batch]` | `(seq_len, input_size)` | one sequence per iteration |
| `actions = a[batch]` (non-recurrent) | `(seq_len,)` -> `(seq_len, 1)` | `.unsqueeze(1)` |
| `actions` (recurrent) | `(seq_len, output_size)` -> `(seq_len, 1, output_size)` | `.view(seq_len, 1, output_size)` |
| `EnvironmentEncoder.embedding(states)` | `(seq_len, embedding_size)` | Linear layer |
| `embed.view(seq_len, 1, -1)` | `(seq_len, 1, embedding_size)` | reshaped for LSTM `batch_first=False` |
| `LSTM output` | `(seq_len, 1, hidden_size)` | `batch_first=True` in constructor but view forces `(seq, batch=1, feature)` |
| `meta_layer(lstm_out)` | `(seq_len, 1, meta_param_size)` | encoding fed to DDPG actor/critic |
| `target_actor(states_)` | `(seq_len, 1, output_size)` | acting on LSTM encoding |
| `target_critic(states_, target_actions)` | `(seq_len, 1, 1)` | |
| `squeeze(q_value_, -1)` | `(seq_len, 1)` | for target computation |
| `unsqueeze(rewards, 1)` | `(seq_len, 1)` | broadcast to match |
| `target` | `(seq_len, 1)` | Bellman target per timestep |
| `squeeze(q_value, -1)` | `(seq_len, 1)` | current critic output |
| `value_loss` | scalar | `MSE(squeeze(q_value,-1), target)` |
| `actor_loss` | scalar | `-mean(critic(states, new_actions))` |
| `batch_loss` | scalar | accumulated sum of per-sequence actor losses |

---

## Attention (A-GSP) Path

Source: `learning_aids.py:341-351`, `gsp_rl/src/networks/self_attention.py`

This is supervised learning, not RL. The `AttentionEncoder` processes a sliding window of observations.

| Stage | Shape | Notes |
|-------|-------|-------|
| `attention_observation` (sliding buffer) | list of `seq_len` observations, each `(gsp_input_size,)` | maintained in `Actor.choose_action` |
| `observation` passed to encoder | `(1, seq_len, gsp_input_size)` | `np.array` then `.unsqueeze(0)` adds batch dim |
| `observations` (training batch) | `(batch_size, seq_len, gsp_input_size)` | from `AttentionSequenceReplayBuffer` |
| `labels` | `(batch_size,)` -> `(batch_size, 1)` | `.unsqueeze(-1)` in loss computation |
| `word_embedding(x)` | `(N, seq_len, 1)` | Linear(`input_size`, `hidden_size`) -> ReLU -> Linear(`hidden_size`, 1) |
| `positions` | `(N, seq_len)` | `arange(seq_len).expand(N, seq_len)` |
| `position_embedding(positions)` | `(N, seq_len, embed_size)` | `nn.Embedding(max_length, embed_size)` |
| `out = word_embedding + position_embedding` | `(N, seq_len, embed_size)` | word embedding `(N, seq_len, 1)` broadcast to `embed_size` |
| `TransformerBlock output (mp)` | `(N, seq_len, embed_size)` | self-attention + feed-forward + layer norm |
| `mp.view(N, -1)` | `(N, seq_len * embed_size)` | flattened for final linear |
| `fc_out(mp.view(N,-1))` | `(N, output_size)` | `Linear(embed_size * max_length, output_size)` |
| `tanh(fc_out) * min_max_action` | `(N, output_size)` | final prediction in `[-min_max_action, min_max_action]` |
| `loss = MSE(pred_headings, labels.unsqueeze(-1))` | scalar | supervised MSE |

---

## GSP Augmentation (All Paths)

When `GSP=True`, the GSP prediction is concatenated with the raw observation before the main network sees it.

| Stage | Shape | Notes |
|-------|-------|-------|
| raw `observation` | `(input_size,)` | from environment |
| `gsp_prediction` | `(gsp_output_size,)` | output of `gsp_networks['actor']` |
| augmented observation | `(input_size + gsp_output_size,)` | concatenated; equals `network_input_size` |
| `network_input_size` in all main-network constructors | `input_size + gsp_output_size` | set in `Actor.__init__` when `gsp=True` |
