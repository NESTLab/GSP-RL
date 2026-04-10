"""Shared fixtures for convergence tests — collects rewards and generates plots."""

import os
import json

import numpy as np
import pytest


PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "convergence_plots")


@pytest.fixture(scope="session")
def convergence_plots():
    """Session-scoped dict that collects {name: rewards_list} from each test."""
    return {}


@pytest.fixture(autouse=True, scope="session")
def generate_plots_at_end(convergence_plots):
    """After all convergence tests finish, generate per-model learning curves."""
    yield

    if not convergence_plots:
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Save raw data as JSON for downstream use
    serializable = {k: [float(v) for v in vals] for k, vals in convergence_plots.items()}
    with open(os.path.join(PLOT_DIR, "rewards.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for name, rewards in convergence_plots.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode reward")

        # 10-episode rolling average
        if len(rewards) >= 10:
            rolling = np.convolve(rewards, np.ones(10) / 10, mode="valid")
            ax.plot(np.arange(10, len(rewards) + 1), rolling,
                    color="darkblue", linewidth=2, label="10-ep rolling avg")

        avg_last_20 = np.mean(rewards[-20:])
        ax.axhline(avg_last_20, color="red", linestyle="--", linewidth=1,
                    label=f"Avg last 20: {avg_last_20:.1f}")

        algo, env_name = name.split("_", 1)
        ax.set_title(f"{algo} — {env_name}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{name}.png"), dpi=120)
        plt.close(fig)
