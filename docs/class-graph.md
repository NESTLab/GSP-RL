# GSP-RL Class Graph — Layer 2 Context

Layer 2 (class + method level) context graph for LLM loading. Split into four focused sub-diagrams to stay within context window limits.

---

## Sub-diagram A: Inheritance Chain

```mermaid
classDiagram
    class Hyperparameters {
        +float gamma
        +float tau
        +float alpha
        +float beta
        +float lr
        +float epsilon
        +float eps_min
        +float eps_dec
        +int batch_size
        +int mem_size
        +float noise
        +int update_actor_iter
        +int warmup
        +int time_step
        +int gsp_learning_offset
        +int gsp_batch_size
        +int replace_target_ctr
    }

    class NetworkAids {
        +make_DQN_networks()
        +make_DDQN_networks()
        +make_DDPG_networks()
        +make_TD3_networks()
        +make_RDDPG_networks()
        +make_Attention_Encoder()
        +learn_DQN()
        +learn_DDQN()
        +learn_DDPG()
        +learn_RDDPG()
        +learn_TD3()
        +learn_attention()
        +DQN_DDQN_choose_action()
        +DDPG_choose_action()
        +TD3_choose_action()
        +Attention_choose_action()
        +update_DDPG_network_parameters()
        +update_TD3_network_parameters()
        +sample_memory()
        +sample_attention_memory()
        +store_transition()
        +store_attention_transition()
        +decrement_epsilon()
    }

    class Actor {
        +build_networks()
        +build_gsp_network()
        +build_DDPG_gsp()
        +build_RDDPG_gsp()
        +choose_action()
        +learn()
        +learn_gsp()
        +store_agent_transition()
        +store_gsp_transition()
        +update_network_parameters()
        +replace_target_network()
        +save_model()
        +load_model()
    }

    class CartPoleAgent {
    }

    Hyperparameters <|-- NetworkAids
    NetworkAids <|-- Actor
    Actor <|-- CartPoleAgent
```

---

## Sub-diagram B: Network Composition (RDDPG)

```mermaid
graph TD
    subgraph make_RDDPG_networks
        shared_ee["shared_ee\n(EnvironmentEncoder / LSTM)"]
        ee_target_actor["ee_target_actor\n(EnvironmentEncoder)"]
        ee_target_critic["ee_target_critic\n(EnvironmentEncoder)"]
    end

    subgraph RDDPGActorNetwork
        EE_actor["EnvironmentEncoder (shared_ee)"]
        DDPG_actor["DDPGActorNetwork"]
    end

    subgraph RDDPGCriticNetwork
        EE_critic["EnvironmentEncoder (shared_ee)"]
        DDPG_critic["DDPGCriticNetwork"]
    end

    subgraph TargetActorNetwork
        EE_tgt_actor["EnvironmentEncoder (ee_target_actor)"]
        DDPG_tgt_actor["DDPGActorNetwork"]
    end

    subgraph TargetCriticNetwork
        EE_tgt_critic["EnvironmentEncoder (ee_target_critic)"]
        DDPG_tgt_critic["DDPGCriticNetwork"]
    end

    state["state (observation)"]
    state --> EE_actor --> encoding_actor["encoding"] --> DDPG_actor --> action["action output"]
    state --> EE_critic --> encoding_critic["encoding"] --> DDPG_critic --> q_value["Q-value output"]
    state --> EE_tgt_actor --> enc_ta["encoding"] --> DDPG_tgt_actor --> tgt_action["target action"]
    state --> EE_tgt_critic --> enc_tc["encoding"] --> DDPG_tgt_critic --> tgt_q["target Q-value"]
```

---

## Sub-diagram C: Algorithm-to-Buffer Mapping

```mermaid
graph LR
    DQN["DQN"] --> RB_D["ReplayBuffer (Discrete)"]
    DDQN["DDQN"] --> RB_D

    DDPG["DDPG"] --> RB_C["ReplayBuffer (Continuous)"]
    TD3["TD3"] --> RB_C
    GSP_DDPG["GSP-DDPG"] --> RB_C

    GSP_RDDPG["GSP-RDDPG"] --> SRB["SequenceReplayBuffer"]
    A_GSP["A-GSP"] --> ASRB["AttentionSequenceReplayBuffer"]
```

---

## Sub-diagram D: `networks` Dict Schema

### Main `networks` dict keys per algorithm

| Algorithm | Dict Keys |
|-----------|-----------|
| DQN | `q_eval`, `q_next`, `replay`, `learning_scheme`, `learn_step_counter` |
| DDQN | `q_eval`, `q_next`, `replay`, `learning_scheme`, `learn_step_counter` |
| DDPG | `actor`, `target_actor`, `critic`, `target_critic`, `replay`, `learning_scheme`, `output_size`, `learn_step_counter` |
| TD3 | `actor`, `target_actor`, `critic_1`, `target_critic_1`, `critic_2`, `target_critic_2`, `replay`, `learning_scheme`, `output_size`, `learn_step_counter` |
| RDDPG | Same keys as DDPG (`actor`/`critic` are `RDDPGActorNetwork`/`RDDPGCriticNetwork` wrappers) |

### `gsp_networks` dict keys per GSP variant

| GSP Variant | Dict Keys |
|-------------|-----------|
| DDPG-GSP | `actor`, `target_actor`, `critic`, `target_critic`, `replay` (ReplayBuffer), `learning_scheme`=`'DDPG'`, `output_size`, `learn_step_counter` |
| RDDPG-GSP | Same structure as DDPG-GSP; `learning_scheme`=`'RDDPG'`; `replay` is `SequenceReplayBuffer` |
| A-GSP | `attention`, `replay` (AttentionSequenceReplayBuffer), `learning_scheme`=`'attention'`, `learn_step_counter` |
