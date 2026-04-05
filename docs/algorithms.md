# GSP-RL Algorithm Reference

Source files: `gsp_rl/src/actors/learning_aids.py`, `gsp_rl/src/actors/actor.py`

---

## DQN

**Description.** Deep Q-Network for discrete action spaces. On each learn step, a batch is sampled from `ReplayBuffer`, Q-values for the current states are computed via `q_eval`, and Q-values for next states are computed via the frozen `q_next` target network. Terminal states are zeroed out before computing the Bellman target. MSE loss is computed against `q_pred` (the Q-value at the taken action). After backprop, `learn_step_counter` is incremented and epsilon is decremented.

```
learn_DQN(networks):
    states, actions, rewards, states_, dones = sample_memory(batch_size)
    indices = arange(batch_size)

    q_pred = q_eval(states)[indices, actions]        # (batch_size,)
    q_next = q_next(states_).max(dim=1)              # (batch_size,)
    q_next[dones] = 0.0

    q_target = rewards + gamma * q_next              # (batch_size,)

    loss = MSE(q_target, q_pred)
    loss.backward()
    q_eval.optimizer.step()

    learn_step_counter += 1
    decrement_epsilon()
    return loss.item()
```

**Implementation notes.**
- `replace_target_network()` is called before `learn_DQN` in `Actor.learn()`; it copies `q_eval` weights to `q_next` every `REPLACE_TARGET_COUNTER` steps.
- Action index into `q_eval` output uses `T.LongTensor` cast.
- Source: `learning_aids.py:163-186`.

---

## DDQN

**Description.** Double DQN decouples action selection from action evaluation to reduce overestimation bias. The current `q_eval` network selects which action has the highest value in the next state (`max_actions`), but the frozen `q_next` network provides the actual value at that action. Everything else mirrors DQN exactly.

```
learn_DDQN(networks):
    states, actions, rewards, states_, dones = sample_memory(batch_size)
    indices = arange(batch_size)

    q_pred = q_eval(states)[indices, actions]        # (batch_size,)
    q_next_vals = q_next(states_)                    # (batch_size, output_size)
    q_eval_next = q_eval(states_)                    # (batch_size, output_size)

    max_actions = argmax(q_eval_next, dim=1)         # action selection by q_eval
    q_next_vals[dones] = 0.0

    q_target = rewards + gamma * q_next_vals[indices, max_actions]

    loss = MSE(q_target, q_pred)
    loss.backward()
    q_eval.optimizer.step()

    learn_step_counter += 1
    decrement_epsilon()
    return loss.item()
```

**Implementation notes.**
- Same `replace_target_network()` guard as DQN applies.
- The key difference from DQN: `q_next_vals[indices, max_actions]` vs `q_next.max(dim=1)`.
- Source: `learning_aids.py:189-217`.

---

## DDPG

**Description.** Deep Deterministic Policy Gradient for continuous action spaces. A critic is trained first using a Bellman target formed from the frozen `target_actor` and `target_critic`. An actor is then updated by ascending the critic's gradient. Target networks are updated via Polyak averaging (`update_network_parameters`) after each `learn()` call.

```
learn_DDPG(networks):
    states, actions, rewards, states_, dones = sample_memory(batch_size)

    # Critic update
    target_actions = target_actor(states_)           # (batch_size, output_size)
    q_value_ = target_critic(states_, target_actions) # (batch_size, 1)
    target = unsqueeze(rewards, 1) + gamma * q_value_ # (batch_size, 1)

    q_value = critic(states, actions)                # (batch_size, 1)
    value_loss = MSE(q_value, target)
    value_loss.backward()
    critic.optimizer.step()

    # Actor update
    new_actions = actor(states)
    actor_loss = -mean(critic(states, new_actions))
    actor_loss.backward()
    actor.optimizer.step()

    learn_step_counter += 1
    return actor_loss.item()
```

**Implementation notes.**
- `rewards` is 1-D `(batch_size,)`; it is unsqueezed to `(batch_size, 1)` to match the critic output shape before computing `target`.
- Critic input is state-action concatenated along `dim=-1` inside `DDPGCriticNetwork.forward`.
- Soft target updates happen in `Actor.learn()` via `update_network_parameters(tau)` before `learn_DDPG` is called.
- Source: `learning_aids.py:219-245`.

---

## TD3

**Description.** Twin Delayed Deep Deterministic Policy Gradient extends DDPG with three improvements: (1) target actions are perturbed with clipped Gaussian noise for smoother value targets; (2) the minimum of two independent critics is used as the Bellman target to combat value overestimation; (3) the actor and target networks are updated less frequently than the critics (every `update_actor_iter` critic steps).

```
learn_TD3(networks):
    states, actions, rewards, states_, dones = sample_memory(batch_size)

    # Noisy target actions
    target_actions = target_actor(states_)
    target_actions += clamp(Normal(0, 0.2), -0.5, 0.5)
    target_actions = clamp(target_actions, -min_max_action, min_max_action)

    # Clipped double-Q target
    q1_ = target_critic_1(states_, target_actions).view(-1) # (batch_size,)
    q2_ = target_critic_2(states_, target_actions).view(-1) # (batch_size,)
    q1_[dones] = 0.0; q2_[dones] = 0.0
    critic_value_ = min(q1_, q2_)
    target = rewards + gamma * critic_value_

    q1 = critic_1(states, actions).squeeze()         # (batch_size,)
    q2 = critic_2(states, actions).squeeze()         # (batch_size,)
    critic_loss = MSE(target, q1) + MSE(target, q2)
    critic_loss.backward()
    critic_1.optimizer.step(); critic_2.optimizer.step()

    learn_step_counter += 1

    # Delayed actor update
    if learn_step_counter % update_actor_iter != 0:
        return 0, 0

    actor_loss = -mean(critic_1(states, actor(states)))
    actor_loss.backward()
    actor.optimizer.step()
    return actor_loss.item()
```

**Implementation notes.**
- Returns `(actor_loss.item(),)` (1-tuple) when the actor updated, `(0, 0)` when skipped. Callers must handle both return shapes.
- Noise scale is hardcoded to `0.2` with clip bounds `[-0.5, 0.5]` inside `learn_TD3`; this is separate from the exploration `NOISE` hyperparameter used at action-selection time.
- Source: `learning_aids.py:294-339`.

---

## RDDPG

**Description.** Recurrent DDPG replaces the flat observation with a sequence processed through `EnvironmentEncoder` (an LSTM with a linear embedding layer). The learn loop is identical to DDPG in structure but iterates over each sequence in the batch individually, accumulating actor loss across sequences. This allows the policy to condition on temporal context rather than a single observation.

```
learn_RDDPG(networks, gsp, recurrent):
    s, a, r, s_, d = sample_memory(batch_size)  # shapes: (batch, seq_len, ...)
    batch_loss = 0

    for batch in range(batch_size):             # loop over sequences
        states  = s[batch]                      # (seq_len, input_size)
        actions = a[batch]
        rewards = r[batch]
        states_ = s_[batch]
        dones   = d[batch]

        # Reshape actions based on recurrent flag
        if not recurrent:
            actions = actions.unsqueeze(1)      # (seq_len, 1, output_size)
        else:
            actions = actions.view(seq_len, 1, output_size)

        # EnvironmentEncoder processes sequences through embed -> LSTM -> meta_layer
        target_actions = target_actor(states_)
        q_value_ = target_critic(states_, target_actions)
        target = unsqueeze(rewards, 1) + gamma * squeeze(q_value_, -1)

        # Critic update
        q_value = critic(states, actions)
        value_loss = MSE(squeeze(q_value, -1), target)
        value_loss.backward()
        critic.optimizer.step()

        # Actor update
        new_actions = actor(states)
        actor_loss = -mean(critic(states, new_actions))
        actor_loss.backward()
        batch_loss += actor_loss.item()
        actor.optimizer.step()

        learn_step_counter += 1

    return batch_loss
```

**Implementation notes.**
- `EnvironmentEncoder.forward`: input `(seq_len, input_size)` -> `embedding` linear -> `(seq_len, embedding_size)` -> `LSTM` with `view(seq_len, 1, -1)` -> `(seq_len, 1, hidden_size)` -> `meta_layer` -> `(seq_len, 1, meta_param_size)`.
- The actor and critic in `RDDPG_networks` are `RDDPGActorNetwork` and `RDDPGCriticNetwork` which wrap a shared `EnvironmentEncoder` and a `DDPGActorNetwork`/`DDPGCriticNetwork` respectively.
- When used for GSP, `batch_size` is `gsp_batch_size`; otherwise the main `batch_size`.
- Source: `learning_aids.py:247-292`.

---

## Attention Learning (A-GSP)

**Description.** The attention variant of GSP uses a supervised learning setup rather than RL. The `AttentionEncoder` is a transformer encoder that accepts a sliding window of past observations and predicts the next global state value. The replay buffer stores `(observation_sequence, label)` pairs rather than SARSD tuples. Loss is MSE between the predicted value and the ground-truth label.

```
learn_attention(networks):
    if replay.mem_ctr < gsp_batch_size:
        return 0
    observations, labels = sample_attention_memory(batch_size)
    # observations: (batch_size, seq_len, gsp_input_size)
    # labels: (batch_size,)

    pred_headings = attention(observations)          # (batch_size, output_size)
    loss = MSE(pred_headings, labels.unsqueeze(-1))  # labels -> (batch_size, 1)
    loss.backward()
    attention.optimizer.step()

    learn_step_counter += 1
    return loss.item()
```

**Implementation notes.**
- This is purely supervised; there are no actor/critic networks, no replay of (s, a, r, s_, done).
- `AttentionEncoder.forward` pipeline: `word_embedding` (Linear -> ReLU -> Linear, output `(N, seq_len, 1)`) broadcast-added to `position_embedding` `(N, seq_len, embed_size)` -> `TransformerBlock` -> `fc_out(mp.view(N, -1))` -> `tanh * min_max_action`.
- The position embedding output dominates the word embedding (size `embed_size` vs `1`); the word embedding provides a scalar modulation.
- Source: `learning_aids.py:341-351`.

---

## GSP Integration

**Description.** Global State Prediction (GSP) adds a second set of networks (`gsp_networks`) that predict some hidden global state. Their prediction is concatenated with the main observation to form an augmented input for the primary policy network, so the policy implicitly conditions on predicted global context. GSP learning is triggered inside `Actor.learn()` every `gsp_learning_offset` (= `GSP_LEARNING_FREQUENCY`) main-network learn steps.

```
Actor.learn():
    if replay.mem_ctr < batch_size:
        return

    # GSP learning hook
    if gsp and (learn_step_counter % gsp_learning_offset == 0):
        learn_gsp()

    # Main network learning (DQN / DDQN / DDPG / TD3)
    ...

Actor.learn_gsp():
    if gsp_replay.mem_ctr < gsp_batch_size:
        return
    dispatch to:
        learn_DDPG(gsp_networks)     # if gsp_networks.learning_scheme == 'DDPG'
        learn_RDDPG(gsp_networks)    # if 'RDDPG'
        learn_TD3(gsp_networks)      # if 'TD3'
        learn_attention(gsp_networks) # if 'attention'
```

**Implementation notes.**
- `network_input_size = input_size + gsp_output_size` when `gsp=True`; the GSP prediction is concatenated at the observation level before passing to the main actor/critic.
- GSP network variants mirror main network variants: `DDPG`, `RDDPG`, `TD3`, and `attention` (A-GSP).
- A-GSP uses `AttentionSequenceReplayBuffer`; RDDPG-GSP uses `SequenceReplayBuffer`; others use standard `ReplayBuffer`.
- Source: `actor.py:349-386`.
