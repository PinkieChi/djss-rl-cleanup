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
    dataset_paths: list[str | Path] | None = None,
    validation_dataset_paths: list[str | Path] | None = None,
    validation_every: int | None = None,
    reward_mode: str = "sharp",
    gamma: float = 0.99,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    learning_rate: float = 0.001,
    train_start: int = 1000,
    output_dir: str | Path = "outputs/training",
    seed: int | None = None,
    use_wandb: bool = False,
    return_metadata: bool = False,
) -> float | dict[str, object]:
    """Train the DQN agent after initializing notebook-era globals explicitly."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    resolved_dataset_path = Path(dataset_path).resolve() if dataset_path is not None else None
    resolved_dataset_paths = [Path(path).resolve() for path in dataset_paths] if dataset_paths else None
    resolved_validation_dataset_paths = (
        [Path(path).resolve() for path in validation_dataset_paths] if validation_dataset_paths else None
    )
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    env = make_env(
        dataset_path=resolved_dataset_paths[0] if resolved_dataset_paths else resolved_dataset_path,
        reward_mode=reward_mode,
    )
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
            "validation_every": validation_every,
            "dataset_path": str(resolved_dataset_path) if resolved_dataset_path is not None else None,
            "dataset_paths": [str(path) for path in resolved_dataset_paths] if resolved_dataset_paths is not None else None,
            "validation_dataset_paths": (
                [str(path) for path in resolved_validation_dataset_paths]
                if resolved_validation_dataset_paths is not None
                else None
            ),
            "reward_mode": reward_mode,
            "gamma": gamma,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "learning_rate": learning_rate,
            "train_start": train_start,
            "output_dir": str(output_path),
            "seed": seed,
        },
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(output_path)
        result = core.train_agents(
            hidden_layers,
            neurons_per_layer or DEFAULT_NEURONS_PER_LAYER,
            batch_size,
            episodes=episodes,
            eval_every=eval_every,
            dataset_path=str(resolved_dataset_path) if resolved_dataset_path is not None else None,
            dataset_paths=[str(path) for path in resolved_dataset_paths] if resolved_dataset_paths is not None else None,
            validation_dataset_paths=(
                [str(path) for path in resolved_validation_dataset_paths]
                if resolved_validation_dataset_paths is not None
                else None
            ),
            validation_every=validation_every,
            return_metadata=return_metadata,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            learning_rate=learning_rate,
            train_start=train_start,
        )
        if isinstance(result, dict):
            checkpoint_filename = str(result.get("checkpoint_filename", ""))
            result["checkpoint_path"] = str(output_path / checkpoint_filename) if checkpoint_filename else None
            result["training_history_path"] = str(output_path / "training_history.csv")
        return result
    finally:
        os.chdir(previous_cwd)
        run.finish()
