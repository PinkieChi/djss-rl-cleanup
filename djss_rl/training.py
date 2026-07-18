"""Training entrypoints for the extracted DJSS RL code."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb

from . import core
from .environment import DEFAULT_DATASET, make_env
from .evaluation import DEFAULT_BATCH_SIZE, DEFAULT_HIDDEN_LAYERS, DEFAULT_NEURONS_PER_LAYER


def train_agents(
    *,
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
    neurons_per_layer: list[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    episodes: int = 1000,
    eval_every: int = 25,
    dataset_path: str | Path | None = DEFAULT_DATASET,
    output_dir: str | Path = "outputs/training",
    seed: int | None = None,
    use_wandb: bool = False,
) -> float:
    """Train the DQN agent after initializing notebook-era globals explicitly."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    resolved_dataset_path = Path(dataset_path).resolve() if dataset_path is not None else None
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    env = make_env(dataset_path=resolved_dataset_path)
    core.state_dim = env.observation_space.shape[0]
    core.output_dim = env.action_space.n
    wandb_mode = os.getenv("WANDB_MODE", "offline") if use_wandb else "disabled"

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "djss-rl"),
        mode=wandb_mode,
        config={
            "hidden_layers": hidden_layers,
            "neurons_per_layer": neurons_per_layer or DEFAULT_NEURONS_PER_LAYER,
            "batch_size": batch_size,
            "episodes": episodes,
            "eval_every": eval_every,
            "dataset_path": str(resolved_dataset_path) if resolved_dataset_path is not None else None,
            "output_dir": str(output_path),
            "seed": seed,
        },
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(output_path)
        return core.train_agents(
            hidden_layers,
            neurons_per_layer or DEFAULT_NEURONS_PER_LAYER,
            batch_size,
            episodes=episodes,
            eval_every=eval_every,
            dataset_path=str(resolved_dataset_path) if resolved_dataset_path is not None else None,
        )
    finally:
        os.chdir(previous_cwd)
        run.finish()
