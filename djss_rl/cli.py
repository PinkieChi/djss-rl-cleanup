"""Command line entrypoints for the cleaned DJSS RL project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .environment import DEFAULT_DATASET, make_env
from .evaluation import DEFAULT_CHECKPOINT, evaluate_all
from .training import train_agents


def _print_env_summary(env, *, mode: str) -> None:
    world = env.world
    print(f"\n{mode.upper()}_RUN_OK")
    print("jobs", len(world.jobs))
    print("machines", len(world.machines))
    print("operations", world.operations)
    print("total_timestamp", world.total_timestamp)
    print("observation_shape", env.observation_space.shape)
    print("action_space", env.action_space.n)


def _print_result(result) -> None:
    print(
        "{}: Tardiness rate: {}, makespan: {} hours {} minutes, mean Machine Utilization: {}".format(
            result.name,
            result.tardiness_rate,
            int(result.makespan) // 60,
            int(result.makespan) % 60,
            result.mean_machine_utilization,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cleaned DJSS RL project safely.")
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory containing the notebook and dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset path relative to the project directory.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="Run setup and environment construction only.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run baseline and checkpoint evaluation without training.")
    evaluate_parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Saved PyTorch checkpoint path.",
    )

    train_parser = subparsers.add_parser("train", help="Train from scratch.")
    train_parser.add_argument("--episodes", type=int, default=1, help="Number of training episodes.")
    train_parser.add_argument("--eval-every", type=int, default=25, help="Evaluation cadence retained for compatibility.")
    train_parser.add_argument("--output-dir", default="outputs/training", help="Directory for generated training checkpoints.")
    train_parser.add_argument("--wandb", action="store_true", help="Initialize an offline W&B run for training logs.")

    args = parser.parse_args()
    project_dir = Path(args.project_dir)
    dataset_path = project_dir / args.dataset

    if args.command == "smoke":
        env = make_env(dataset_path=dataset_path)
        _print_env_summary(env, mode="smoke")
    elif args.command == "evaluate":
        results = evaluate_all(dataset_path=dataset_path, checkpoint_path=project_dir / args.checkpoint)
        for result in results:
            _print_result(result)
        print("\nEVALUATION_RUN_OK")
        print("costs", [result.tardiness_rate for result in results])
    elif args.command == "train":
        score = train_agents(
            episodes=args.episodes,
            eval_every=args.eval_every,
            dataset_path=dataset_path,
            output_dir=project_dir / args.output_dir,
            use_wandb=args.wandb,
        )
        print("\nTRAIN_RUN_OK")
        print("best_score", score)


if __name__ == "__main__":
    main()
