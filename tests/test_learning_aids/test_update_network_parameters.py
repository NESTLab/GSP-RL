import torch as T
import pytest
from gsp_rl.src.actors.actor import Actor

CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.5,
    "ALPHA": 0.001,
    "BETA": 0.002,
    "LR": 0.001,
    "EPSILON": 0.0,
    "EPS_MIN": 0.0,
    "EPS_DEC": 0.0,
    "BATCH_SIZE": 8,
    "MEM_SIZE": 100,
    "REPLACE_TARGET_COUNTER": 10,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 1,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": 8,
}


def make_ddpg_actor():
    return Actor(
        id=1,
        config=CONFIG,
        network="DDPG",
        input_size=8,
        output_size=2,
        min_max_action=1.0,
        meta_param_size=1,
    )


def make_td3_actor():
    return Actor(
        id=1,
        config=CONFIG,
        network="TD3",
        input_size=8,
        output_size=2,
        min_max_action=1.0,
        meta_param_size=1,
    )


# ---------------------------------------------------------------------------
# DDPG tests
# ---------------------------------------------------------------------------

def test_ddpg_soft_update_tau_half():
    """tau=0.5: target_after = 0.5 * online + 0.5 * target_before for actor and critic."""
    actor = make_ddpg_actor()
    networks = actor.networks
    tau = 0.5

    # Snapshot target and online params before update
    target_actor_before = {n: p.data.clone() for n, p in networks["target_actor"].named_parameters()}
    target_critic_before = {n: p.data.clone() for n, p in networks["target_critic"].named_parameters()}
    online_actor = {n: p.data.clone() for n, p in networks["actor"].named_parameters()}
    online_critic = {n: p.data.clone() for n, p in networks["critic"].named_parameters()}

    actor.update_DDPG_network_parameters(tau, networks)

    for name, p in networks["target_actor"].named_parameters():
        expected = tau * online_actor[name] + (1 - tau) * target_actor_before[name]
        T.testing.assert_close(p.data, expected, atol=1e-6, rtol=1e-6)

    for name, p in networks["target_critic"].named_parameters():
        expected = tau * online_critic[name] + (1 - tau) * target_critic_before[name]
        T.testing.assert_close(p.data, expected, atol=1e-6, rtol=1e-6)


def test_ddpg_soft_update_tau_zero_no_change():
    """tau=0: target params are unchanged after update."""
    actor = make_ddpg_actor()
    networks = actor.networks
    tau = 0.0

    target_actor_before = {n: p.data.clone() for n, p in networks["target_actor"].named_parameters()}
    target_critic_before = {n: p.data.clone() for n, p in networks["target_critic"].named_parameters()}

    actor.update_DDPG_network_parameters(tau, networks)

    for name, p in networks["target_actor"].named_parameters():
        T.testing.assert_close(p.data, target_actor_before[name], atol=1e-6, rtol=1e-6)

    for name, p in networks["target_critic"].named_parameters():
        T.testing.assert_close(p.data, target_critic_before[name], atol=1e-6, rtol=1e-6)


def test_ddpg_soft_update_tau_one_full_copy():
    """tau=1: target params equal online params after update."""
    actor = make_ddpg_actor()
    networks = actor.networks
    tau = 1.0

    online_actor = {n: p.data.clone() for n, p in networks["actor"].named_parameters()}
    online_critic = {n: p.data.clone() for n, p in networks["critic"].named_parameters()}

    actor.update_DDPG_network_parameters(tau, networks)

    for name, p in networks["target_actor"].named_parameters():
        T.testing.assert_close(p.data, online_actor[name], atol=1e-6, rtol=1e-6)

    for name, p in networks["target_critic"].named_parameters():
        T.testing.assert_close(p.data, online_critic[name], atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# TD3 tests
# ---------------------------------------------------------------------------

def test_td3_soft_update_all_three_targets():
    """tau=0.5: all three TD3 target networks updated with Polyak averaging."""
    actor = make_td3_actor()
    networks = actor.networks
    tau = 0.5

    # Snapshot target params before update
    target_actor_before = {n: p.data.clone() for n, p in networks["target_actor"].named_parameters()}
    target_critic_1_before = {n: p.data.clone() for n, p in networks["target_critic_1"].named_parameters()}
    target_critic_2_before = {n: p.data.clone() for n, p in networks["target_critic_2"].named_parameters()}

    # Snapshot online params before update
    online_actor = {n: p.data.clone() for n, p in networks["actor"].named_parameters()}
    online_critic_1 = {n: p.data.clone() for n, p in networks["critic_1"].named_parameters()}
    online_critic_2 = {n: p.data.clone() for n, p in networks["critic_2"].named_parameters()}

    actor.update_TD3_network_parameters(tau, networks)

    for name, p in networks["target_actor"].named_parameters():
        expected = tau * online_actor[name] + (1 - tau) * target_actor_before[name]
        T.testing.assert_close(p.data, expected, atol=1e-6, rtol=1e-6)

    for name, p in networks["target_critic_1"].named_parameters():
        expected = tau * online_critic_1[name] + (1 - tau) * target_critic_1_before[name]
        T.testing.assert_close(p.data, expected, atol=1e-6, rtol=1e-6)

    for name, p in networks["target_critic_2"].named_parameters():
        expected = tau * online_critic_2[name] + (1 - tau) * target_critic_2_before[name]
        T.testing.assert_close(p.data, expected, atol=1e-6, rtol=1e-6)
