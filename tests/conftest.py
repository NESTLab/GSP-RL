"""Shared fixtures and configuration for GSP-RL test suite."""

import os
import yaml
import pytest
import numpy as np
import torch as T


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: convergence tests (run with -m slow)")


@pytest.fixture
def sample_config():
    """Minimal config dict for Actor construction."""
    return {
        "GAMMA": 0.99,
        "TAU": 0.005,
        "ALPHA": 0.001,
        "BETA": 0.002,
        "LR": 0.0001,
        "EPSILON": 1.0,
        "EPS_MIN": 0.01,
        "EPS_DEC": 0.001,
        "BATCH_SIZE": 32,
        "MEM_SIZE": 1000,
        "REPLACE_TARGET_COUNTER": 100,
        "NOISE": 0.1,
        "UPDATE_ACTOR_ITER": 2,
        "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100,
        "GSP_BATCH_SIZE": 16,
    }


@pytest.fixture
def device():
    return "cuda:0" if T.cuda.is_available() else "cpu"
