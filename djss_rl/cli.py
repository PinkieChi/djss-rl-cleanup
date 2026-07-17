"""Command line entrypoints for the cleaned DJSS RL project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .notebook_runner import DEFAULT_DATASET, execute_notebook


def _print_summary(summary, *, mode: str) -> None:
    print(f"\n{mode.upper()}_RUN_OK")
    print("jobs", summary.jobs)
    print("machines", summary.machines)
    print("operations", summary.operations)
    print("total_timestamp", summary.total_timestamp)
    print("observation_shape", summary.observation_shape)
    print("action_space", summary.action_space)
    if summary.costs:
        print("costs", summary.costs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cleaned DJSS RL notebook safely.")
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
    subparsers.add_parser("evaluate", help="Run baseline and checkpoint evaluation without training.")

    args = parser.parse_args()
    project_dir = Path(args.project_dir)

    if args.command == "smoke":
        summary = execute_notebook(project_dir, dataset_path=args.dataset)
        _print_summary(summary, mode="smoke")
    elif args.command == "evaluate":
        summary = execute_notebook(project_dir, dataset_path=args.dataset, run_evaluation=True)
        _print_summary(summary, mode="evaluation")


if __name__ == "__main__":
    main()
