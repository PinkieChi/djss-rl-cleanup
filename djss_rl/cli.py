"""Command line entrypoints for the cleaned DJSS RL project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import DatasetSpec, generate_dataset
from .environment import DEFAULT_DATASET, make_env
from .evaluation import DEFAULT_CHECKPOINT, evaluate_all
from .experiments import run_baseline_grid, run_rl_generalization_study
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


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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
    train_parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible training.")
    train_parser.add_argument("--wandb", action="store_true", help="Initialize an offline W&B run for training logs.")

    generate_parser = subparsers.add_parser("generate-dataset", help="Generate a stable-ID dataset .ini file.")
    generate_parser.add_argument("--output", required=True, help="Destination .ini path.")
    generate_parser.add_argument("--jobs", type=int, default=50)
    generate_parser.add_argument("--work-centers", type=int, default=5)
    generate_parser.add_argument("--machines-per-work-center", type=int, default=3)
    generate_parser.add_argument("--ddt", type=float, default=0.5)
    generate_parser.add_argument("--arrival-rate", type=int, default=50)
    generate_parser.add_argument("--seed", type=int, default=101)
    generate_parser.add_argument("--initial-job-fraction", type=float, default=0.5)
    generate_parser.add_argument("--min-operations", type=int, default=6)
    generate_parser.add_argument("--max-operations", type=int, default=10)
    generate_parser.add_argument("--min-processing-time", type=int, default=60)
    generate_parser.add_argument("--max-processing-time", type=int, default=120)

    experiment_parser = subparsers.add_parser("experiment", help="Run a generated-instance baseline experiment matrix.")
    experiment_parser.add_argument("--output-dir", default="outputs/experiment-matrix", help="Directory for datasets and result files.")
    experiment_parser.add_argument("--jobs-values", default="20", help="Comma-separated job counts.")
    experiment_parser.add_argument("--ddt-values", default="0.5,1.0,1.5", help="Comma-separated due-date tightness values.")
    experiment_parser.add_argument("--arrival-rates", default="50,100,200", help="Comma-separated arrival-rate values.")
    experiment_parser.add_argument("--seeds", default="101,202,303", help="Comma-separated instance-generation seeds.")
    experiment_parser.add_argument("--work-centers", type=int, default=5)
    experiment_parser.add_argument("--machines-per-work-center", type=int, default=3)
    experiment_parser.add_argument("--initial-job-fraction", type=float, default=0.5)
    experiment_parser.add_argument("--min-operations", type=int, default=6)
    experiment_parser.add_argument("--max-operations", type=int, default=10)
    experiment_parser.add_argument("--min-processing-time", type=int, default=60)
    experiment_parser.add_argument("--max-processing-time", type=int, default=120)
    experiment_parser.add_argument("--include-checkpoint", action="store_true", help="Also evaluate the saved DQN checkpoint.")
    experiment_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Saved PyTorch checkpoint path.")

    rl_parser = subparsers.add_parser("rl-study", help="Train DQN on generated instances and evaluate held-out generalization.")
    rl_parser.add_argument("--output-dir", default="outputs/rl-generalization", help="Directory for generated data, checkpoints, and results.")
    rl_parser.add_argument("--jobs-values", default="20", help="Comma-separated job counts.")
    rl_parser.add_argument("--ddt-values", default="0.5,1.0", help="Comma-separated due-date tightness values.")
    rl_parser.add_argument("--arrival-rates", default="50,100", help="Comma-separated arrival-rate values.")
    rl_parser.add_argument("--train-instance-seeds", default="101,202", help="Comma-separated dataset seeds for training instances.")
    rl_parser.add_argument("--test-instance-seeds", default="303,404", help="Comma-separated dataset seeds for held-out test instances.")
    rl_parser.add_argument("--training-seeds", default="11,22,33", help="Comma-separated random seeds for DQN training.")
    rl_parser.add_argument("--episodes", type=int, default=1000)
    rl_parser.add_argument("--work-centers", type=int, default=5)
    rl_parser.add_argument("--machines-per-work-center", type=int, default=3)
    rl_parser.add_argument("--initial-job-fraction", type=float, default=0.5)
    rl_parser.add_argument("--min-operations", type=int, default=6)
    rl_parser.add_argument("--max-operations", type=int, default=10)
    rl_parser.add_argument("--min-processing-time", type=int, default=60)
    rl_parser.add_argument("--max-processing-time", type=int, default=120)

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
            seed=args.seed,
            use_wandb=args.wandb,
        )
        print("\nTRAIN_RUN_OK")
        print("best_score", score)
    elif args.command == "generate-dataset":
        generated_path = generate_dataset(
            project_dir / args.output,
            spec=DatasetSpec(
                jobs=args.jobs,
                work_centers=args.work_centers,
                machines_per_work_center=args.machines_per_work_center,
                ddt=args.ddt,
                arrival_rate=args.arrival_rate,
                initial_job_fraction=args.initial_job_fraction,
                min_operations=args.min_operations,
                max_operations=args.max_operations,
                min_processing_time=args.min_processing_time,
                max_processing_time=args.max_processing_time,
            ),
            seed=args.seed,
        )
        print("\nDATASET_GENERATION_OK")
        print("dataset_path", generated_path)
    elif args.command == "experiment":
        csv_path, summary_path = run_baseline_grid(
            output_dir=project_dir / args.output_dir,
            jobs_values=_parse_int_list(args.jobs_values),
            ddt_values=_parse_float_list(args.ddt_values),
            arrival_rates=_parse_int_list(args.arrival_rates),
            seeds=_parse_int_list(args.seeds),
            work_centers=args.work_centers,
            machines_per_work_center=args.machines_per_work_center,
            initial_job_fraction=args.initial_job_fraction,
            min_operations=args.min_operations,
            max_operations=args.max_operations,
            min_processing_time=args.min_processing_time,
            max_processing_time=args.max_processing_time,
            include_checkpoint=args.include_checkpoint,
            checkpoint_path=project_dir / args.checkpoint,
        )
        print("\nEXPERIMENT_RUN_OK")
        print("results_csv", csv_path)
        print("summary_markdown", summary_path)
    elif args.command == "rl-study":
        csv_path, summary_path = run_rl_generalization_study(
            output_dir=project_dir / args.output_dir,
            jobs_values=_parse_int_list(args.jobs_values),
            ddt_values=_parse_float_list(args.ddt_values),
            arrival_rates=_parse_int_list(args.arrival_rates),
            train_instance_seeds=_parse_int_list(args.train_instance_seeds),
            test_instance_seeds=_parse_int_list(args.test_instance_seeds),
            training_seeds=_parse_int_list(args.training_seeds),
            episodes=args.episodes,
            work_centers=args.work_centers,
            machines_per_work_center=args.machines_per_work_center,
            initial_job_fraction=args.initial_job_fraction,
            min_operations=args.min_operations,
            max_operations=args.max_operations,
            min_processing_time=args.min_processing_time,
            max_processing_time=args.max_processing_time,
        )
        print("\nRL_STUDY_RUN_OK")
        print("results_csv", csv_path)
        print("summary_markdown", summary_path)


if __name__ == "__main__":
    main()
