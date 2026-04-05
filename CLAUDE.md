# GSP-RL

PyTorch library for deep RL with Global State Prediction (GSP) for decentralized multi-agent swarm robotics. Implements DQN, DDQN, DDPG, TD3, RDDPG with three GSP variants: DDPG-GSP, RDDPG-GSP, and A-GSP (attention-based).

## Quick Orientation

- **Entry point**: `gsp_rl/src/actors/actor.py` -- the `Actor` class is the main public API
- **Inheritance**: `Actor` -> `NetworkAids` -> `Hyperparameters` (all in `actors/`)
- **Config**: All hyperparameters flow through a YAML config dict passed to `Actor.__init__()` -> `Hyperparameters.__init__()`
- **Network storage**: `self.networks` and `self.gsp_networks` are **plain dicts** (not classes), keyed by strings like `'actor'`, `'critic'`, `'q_eval'`, `'replay'`, `'learning_scheme'`, `'learn_step_counter'`
- **GSP augmentation**: When GSP is enabled, `network_input_size = input_size + gsp_output_size`. The GSP prediction is concatenated with the observation before feeding the main action network.

## Context Loading Guide

Load only the docs relevant to your current task:

| Task | Load these files |
|------|-----------------|
| Understand overall architecture | `docs/architecture.md` |
| Refactor inheritance or add algorithm family | `docs/architecture.md` + `docs/class-graph.md` |
| Add/modify a neural network | `docs/modules/networks.md` + `docs/data-flow.md` |
| Add/modify a replay buffer | `docs/modules/buffers.md` |
| Debug learning logic or loss computation | `docs/modules/actors.md` + `docs/algorithms.md` |
| Modify hyperparameters or config | `docs/configuration.md` |
| Work on GSP prediction pipeline | `docs/algorithms.md` + `docs/class-graph.md` |
| Debug tensor shape mismatches | `docs/data-flow.md` |

## Project Structure

```
gsp_rl/src/
  actors/
    actor.py           # Actor class -- main agent, builds networks + buffers
    learning_aids.py   # NetworkAids (factory + learn methods) + Hyperparameters
  networks/
    dqn.py             # DQN (discrete, nn.Module)
    ddqn.py            # DDQN (discrete, nn.Module)
    ddpg.py            # DDPGActorNetwork + DDPGCriticNetwork + fanin_init
    td3.py             # TD3ActorNetwork + TD3CriticNetwork
    rddpg.py           # RDDPGActorNetwork + RDDPGCriticNetwork (compose EE + DDPG)
    lstm.py            # EnvironmentEncoder (LSTM-based)
    self_attention.py  # SelfAttention, TransformerBlock, AttentionEncoder
  buffers/
    replay.py          # ReplayBuffer (standard SARSD, circular)
    sequential.py      # SequenceReplayBuffer (two-stage: seq staging -> main)
    attention_sequential.py  # AttentionSequenceReplayBuffer (obs + label)
  utility/
    zmq_utility.py     # ZMQ binary message parsing for CoppeliaSim robotics
examples/baselines/    # CartPole, LunarLander, Pendulum training scripts
tests/                 # Shape validation tests for all network types
```

## Conventions

- Networks stored in plain dicts, not wrapper classes. Dict schema varies by algorithm (see `docs/class-graph.md`).
- All `nn.Module` subclasses self-assign device (`cuda:0` or `cpu`) in their constructors.
- Replay buffers use circular indexing: `mem_ctr % mem_size`.
- The `intention` parameter on `save_checkpoint`/`load_checkpoint` refers to the GSP network (historical naming from "intention prediction").
- `fanin_init` in `ddpg.py` is used for weight initialization in DDPG networks only. TD3 uses default PyTorch init.
- All optimizers are Adam with `weight_decay=1e-4`.
- Tests validate network input/output tensor shapes, not learning convergence.
- Examples subclass or directly instantiate `Actor`, not `NetworkAids`.

## Known Gotchas

- **GSP always builds** (`actor.py:89`): `if gsp is not None` is always True since `gsp` defaults to `False` (and `False is not None`). This means `build_gsp_network('DDPG')` is always called.
- **RDDPG dead code** (`actor.py:159`): `self.build_RDDPG()` is called with 0 args but the method signature (`actor.py:200`) requires 3 positional args. This code path will raise TypeError if reached.
- **`load_stat_dict` typo**: Several `load_checkpoint` methods in the CPU branch call `self.load_stat_dict()` (missing 'e') instead of `self.load_state_dict()`. Affects: `dqn.py:67`, `ddqn.py:66`, `ddpg.py:162`, `td3.py:69`, `lstm.py:73`, `self_attention.py:171`.
- **TD3 vs DDPG cat dim**: `TD3CriticNetwork.forward` uses `dim=1` for state-action concatenation while `DDPGCriticNetwork.forward` uses `dim=-1`. Functionally equivalent for 2D tensors but inconsistent.
- **RDDPG shared encoder**: In `make_RDDPG_networks` (`learning_aids.py:74-82`), actor and critic share one `EnvironmentEncoder` instance, but target networks get separate instances. This is intentional for gradient flow isolation.
- **Module-level `Loss`**: `learning_aids.py:20` defines `Loss = nn.MSELoss()` as a module-level singleton shared across all learn methods.
