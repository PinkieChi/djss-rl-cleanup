"""Experiment matrix runners and summary writers."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import statistics

from scipy.stats import wilcoxon

from .datasets import DatasetSpec, generate_dataset
from .evaluation import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_NEURONS_PER_LAYER,
    HEURISTICS,
    SchedulingResult,
    evaluate_checkpoint,
    run_scheduling,
)
from .environment import make_env
from .training import train_agents


@dataclass(frozen=True)
class ExperimentInstance:
    instance_id: str
    dataset_path: Path
    spec: DatasetSpec
    seed: int


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * _std(values) / (len(values) ** 0.5)


def _wilcoxon_p_value(values: list[float]) -> float:
    nonzero_values = [value for value in values if value != 0]
    if len(nonzero_values) < 2:
        return float("nan")
    return float(wilcoxon(nonzero_values).pvalue)


def _result_to_row(instance: ExperimentInstance, result: SchedulingResult) -> dict[str, str]:
    return {
        "instance_id": instance.instance_id,
        "dataset_path": str(instance.dataset_path),
        "seed": str(instance.seed),
        "jobs": str(instance.spec.jobs),
        "machines": str(instance.spec.machines),
        "work_centers": str(instance.spec.work_centers),
        "machines_per_work_center": str(instance.spec.machines_per_work_center),
        "ddt": str(instance.spec.ddt),
        "arrival_rate": str(instance.spec.arrival_rate),
        "method": result.name,
        "tardiness_rate": repr(result.tardiness_rate),
        "makespan": repr(result.makespan),
        "mean_machine_utilization": repr(result.mean_machine_utilization),
        "corrective_maintenance_actions": repr(result.corrective_maintenance_actions),
        "preventive_maintenance_actions": repr(result.preventive_maintenance_actions),
        "energy_consumption": repr(result.energy_consumption),
        "status": "ok",
        "error": "",
    }


def _error_row(instance: ExperimentInstance, *, method: str, error: Exception) -> dict[str, str]:
    return {
        "instance_id": instance.instance_id,
        "dataset_path": str(instance.dataset_path),
        "seed": str(instance.seed),
        "jobs": str(instance.spec.jobs),
        "machines": str(instance.spec.machines),
        "work_centers": str(instance.spec.work_centers),
        "machines_per_work_center": str(instance.spec.machines_per_work_center),
        "ddt": str(instance.spec.ddt),
        "arrival_rate": str(instance.spec.arrival_rate),
        "method": method,
        "tardiness_rate": "nan",
        "makespan": "nan",
        "mean_machine_utilization": "nan",
        "corrective_maintenance_actions": "nan",
        "preventive_maintenance_actions": "nan",
        "energy_consumption": "nan",
        "status": "error",
        "error": str(error),
    }


def build_instances(
    *,
    output_dir: str | Path,
    jobs_values: list[int],
    ddt_values: list[float],
    arrival_rates: list[int],
    seeds: list[int],
    work_centers: int = 5,
    machines_per_work_center: int = 3,
    initial_job_fraction: float = 0.5,
    min_operations: int = 6,
    max_operations: int = 10,
    min_processing_time: int = 60,
    max_processing_time: int = 120,
    min_compatible_machines: int = 1,
    max_compatible_machines: int | None = None,
) -> list[ExperimentInstance]:
    datasets_dir = Path(output_dir) / "datasets"
    instances: list[ExperimentInstance] = []
    for jobs in jobs_values:
        for ddt in ddt_values:
            for arrival_rate in arrival_rates:
                for seed in seeds:
                    spec = DatasetSpec(
                        jobs=jobs,
                        work_centers=work_centers,
                        machines_per_work_center=machines_per_work_center,
                        ddt=ddt,
                        arrival_rate=arrival_rate,
                        initial_job_fraction=initial_job_fraction,
                        min_operations=min_operations,
                        max_operations=max_operations,
                        min_processing_time=min_processing_time,
                        max_processing_time=max_processing_time,
                        min_compatible_machines=min_compatible_machines,
                        max_compatible_machines=max_compatible_machines,
                    )
                    instance_id = f"j{jobs}_m{spec.machines}_ddt{ddt:g}_arr{arrival_rate}_seed{seed}"
                    dataset_path = datasets_dir / f"{instance_id}.ini"
                    generate_dataset(dataset_path, spec=spec, seed=seed)
                    instances.append(ExperimentInstance(instance_id, dataset_path, spec, seed))
    return instances


def run_baseline_grid(
    *,
    output_dir: str | Path,
    jobs_values: list[int],
    ddt_values: list[float],
    arrival_rates: list[int],
    seeds: list[int],
    work_centers: int = 5,
    machines_per_work_center: int = 3,
    initial_job_fraction: float = 0.5,
    min_operations: int = 6,
    max_operations: int = 10,
    min_processing_time: int = 60,
    max_processing_time: int = 120,
    min_compatible_machines: int = 1,
    max_compatible_machines: int | None = None,
    include_checkpoint: bool = False,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
) -> tuple[Path, Path]:
    """Generate datasets, evaluate methods, and write CSV plus Markdown summaries."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    instances = build_instances(
        output_dir=output_path,
        jobs_values=jobs_values,
        ddt_values=ddt_values,
        arrival_rates=arrival_rates,
        seeds=seeds,
        work_centers=work_centers,
        machines_per_work_center=machines_per_work_center,
        initial_job_fraction=initial_job_fraction,
        min_operations=min_operations,
        max_operations=max_operations,
        min_processing_time=min_processing_time,
        max_processing_time=max_processing_time,
        min_compatible_machines=min_compatible_machines,
        max_compatible_machines=max_compatible_machines,
    )
    _write_json(
        output_path / "study_config.json",
        {
            "study_type": "baseline_grid",
            "jobs_values": jobs_values,
            "ddt_values": ddt_values,
            "arrival_rates": arrival_rates,
            "seeds": seeds,
            "work_centers": work_centers,
            "machines_per_work_center": machines_per_work_center,
            "initial_job_fraction": initial_job_fraction,
            "min_operations": min_operations,
            "max_operations": max_operations,
            "min_processing_time": min_processing_time,
            "max_processing_time": max_processing_time,
            "min_compatible_machines": min_compatible_machines,
            "max_compatible_machines": max_compatible_machines,
            "include_checkpoint": include_checkpoint,
            "checkpoint_path": str(checkpoint_path),
        },
    )

    csv_path = output_path / "results.csv"
    rows: list[dict[str, str]] = []
    completed: set[tuple[str, str]] = set()
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        completed = {(row["instance_id"], row["method"]) for row in rows}

    for instance in instances:
        for index, heuristic in enumerate(HEURISTICS):
            if (instance.instance_id, heuristic) in completed:
                continue
            try:
                env = make_env(dataset_path=instance.dataset_path)
                result = run_scheduling(env, env.world, name=heuristic, decision_rule=index)
                rows.append(_result_to_row(instance, result))
            except Exception as exc:  # noqa: BLE001 - experiment rows should capture failures.
                rows.append(_error_row(instance, method=heuristic, error=exc))
            completed.add((instance.instance_id, heuristic))
            _write_rows(csv_path, rows)
        if include_checkpoint:
            method = "Ours"
            if (instance.instance_id, method) not in completed:
                try:
                    result = evaluate_checkpoint(dataset_path=instance.dataset_path, checkpoint_path=checkpoint_path)
                    rows.append(_result_to_row(instance, result))
                except Exception as exc:  # noqa: BLE001 - experiment rows should capture failures.
                    rows.append(_error_row(instance, method=method, error=exc))
                completed.add((instance.instance_id, method))
                _write_rows(csv_path, rows)

    summary_path = output_path / "summary.md"
    summary_path.write_text(make_markdown_summary(rows), encoding="utf-8")
    return csv_path, summary_path


def _write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _experiment_row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("split", "test"),
        row["instance_id"],
        row["method"],
        row.get("training_seed", ""),
    )


def _comparison_stats(
    rows: list[dict[str, str]],
    *,
    method: str,
    baseline: str,
    split: str = "test",
) -> dict[str, object]:
    by_method_instance: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("split", "test") != split or row.get("status", "ok") != "ok":
            continue
        by_method_instance[row["method"]][row["instance_id"]] = row

    differences = []
    wins = losses = ties = 0
    for row in [
        item
        for item in rows
        if item.get("split", "test") == split
        and item.get("status", "ok") == "ok"
        and item["method"] == method
    ]:
        baseline_row = by_method_instance.get(baseline, {}).get(row["instance_id"])
        if baseline_row is None:
            continue
        difference = float(row["tardiness_rate"]) - float(baseline_row["tardiness_rate"])
        differences.append(difference)
        if difference < 0:
            wins += 1
        elif difference > 0:
            losses += 1
        else:
            ties += 1

    return {
        "baseline": baseline,
        "compared_pairs": len(differences),
        "mean_difference": _mean(differences),
        "std_difference": _std(differences),
        "wilcoxon_p": _wilcoxon_p_value(differences),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def _method_metric(rows: list[dict[str, str]], *, method: str, metric: str = "tardiness_rate", split: str = "test") -> float:
    values = [
        float(row[metric])
        for row in rows
        if row.get("split", "test") == split and row.get("status", "ok") == "ok" and row["method"] == method
    ]
    return _mean(values)


def run_rl_generalization_study(
    *,
    output_dir: str | Path,
    jobs_values: list[int],
    ddt_values: list[float],
    arrival_rates: list[int],
    train_instance_seeds: list[int],
    test_instance_seeds: list[int],
    training_seeds: list[int],
    validation_instance_seeds: list[int] | None = None,
    episodes: int = 1000,
    validation_every: int | None = None,
    reward_mode: str = "sharp",
    gamma: float = 0.99,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    learning_rate: float = 0.001,
    train_start: int = 1000,
    work_centers: int = 5,
    machines_per_work_center: int = 3,
    initial_job_fraction: float = 0.5,
    min_operations: int = 6,
    max_operations: int = 10,
    min_processing_time: int = 60,
    max_processing_time: int = 120,
    min_compatible_machines: int = 1,
    max_compatible_machines: int | None = None,
) -> tuple[Path, Path]:
    """Train DQN agents on generated instances and evaluate on held-out instances."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_path / "study_config.json",
        {
            "study_type": "rl_generalization",
            "jobs_values": jobs_values,
            "ddt_values": ddt_values,
            "arrival_rates": arrival_rates,
            "train_instance_seeds": train_instance_seeds,
            "validation_instance_seeds": validation_instance_seeds or [],
            "test_instance_seeds": test_instance_seeds,
            "training_seeds": training_seeds,
            "episodes": episodes,
            "validation_every": validation_every,
            "reward_mode": reward_mode,
            "gamma": gamma,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "learning_rate": learning_rate,
            "train_start": train_start,
            "work_centers": work_centers,
            "machines_per_work_center": machines_per_work_center,
            "initial_job_fraction": initial_job_fraction,
            "min_operations": min_operations,
            "max_operations": max_operations,
            "min_processing_time": min_processing_time,
            "max_processing_time": max_processing_time,
            "min_compatible_machines": min_compatible_machines,
            "max_compatible_machines": max_compatible_machines,
        },
    )
    train_instances = build_instances(
        output_dir=output_path / "train",
        jobs_values=jobs_values,
        ddt_values=ddt_values,
        arrival_rates=arrival_rates,
        seeds=train_instance_seeds,
        work_centers=work_centers,
        machines_per_work_center=machines_per_work_center,
        initial_job_fraction=initial_job_fraction,
        min_operations=min_operations,
        max_operations=max_operations,
        min_processing_time=min_processing_time,
        max_processing_time=max_processing_time,
        min_compatible_machines=min_compatible_machines,
        max_compatible_machines=max_compatible_machines,
    )
    validation_instances = (
        build_instances(
            output_dir=output_path / "validation",
            jobs_values=jobs_values,
            ddt_values=ddt_values,
            arrival_rates=arrival_rates,
            seeds=validation_instance_seeds,
            work_centers=work_centers,
            machines_per_work_center=machines_per_work_center,
            initial_job_fraction=initial_job_fraction,
            min_operations=min_operations,
            max_operations=max_operations,
            min_processing_time=min_processing_time,
            max_processing_time=max_processing_time,
            min_compatible_machines=min_compatible_machines,
            max_compatible_machines=max_compatible_machines,
        )
        if validation_instance_seeds
        else []
    )
    test_instances = build_instances(
        output_dir=output_path / "test",
        jobs_values=jobs_values,
        ddt_values=ddt_values,
        arrival_rates=arrival_rates,
        seeds=test_instance_seeds,
        work_centers=work_centers,
        machines_per_work_center=machines_per_work_center,
        initial_job_fraction=initial_job_fraction,
        min_operations=min_operations,
        max_operations=max_operations,
        min_processing_time=min_processing_time,
        max_processing_time=max_processing_time,
        min_compatible_machines=min_compatible_machines,
        max_compatible_machines=max_compatible_machines,
    )

    csv_path = output_path / "rl_results.csv"
    training_summary_path = output_path / "training_summary.csv"
    rows: list[dict[str, str]] = _read_rows(csv_path)
    completed = {_experiment_row_key(row) for row in rows}
    for split, instances in (("validation", validation_instances), ("test", test_instances)):
        for instance in instances:
            for index, heuristic in enumerate(HEURISTICS):
                row_key = (split, instance.instance_id, heuristic, "")
                if row_key in completed:
                    continue
                try:
                    env = make_env(dataset_path=instance.dataset_path)
                    result = run_scheduling(env, env.world, name=heuristic, decision_rule=index)
                    row = _result_to_row(instance, result)
                except Exception as exc:  # noqa: BLE001 - experiment rows should capture failures.
                    row = _error_row(instance, method=heuristic, error=exc)
                row["split"] = split
                row["training_seed"] = ""
                rows.append(row)
                completed.add(row_key)
                _write_rows(csv_path, rows)

    train_dataset_paths = [instance.dataset_path for instance in train_instances]
    validation_dataset_paths = [instance.dataset_path for instance in validation_instances]
    checkpoint_name = (
        f"Best_agent_hidden_layers_{DEFAULT_HIDDEN_LAYERS}"
        f"neurons_per_layer_{DEFAULT_NEURONS_PER_LAYER}_batch_size_{DEFAULT_BATCH_SIZE}.pth"
    )
    training_summary_rows = _read_rows(training_summary_path)
    training_summary_by_seed = {row["training_seed"]: row for row in training_summary_rows}
    for training_seed in training_seeds:
        agent_output_dir = output_path / "agents" / f"seed-{training_seed}"
        existing_training_row = training_summary_by_seed.get(str(training_seed))
        existing_checkpoint = (
            Path(existing_training_row["checkpoint_path"])
            if existing_training_row and existing_training_row.get("checkpoint_path")
            else None
        )
        if existing_training_row and existing_checkpoint and existing_checkpoint.exists():
            checkpoint_path = existing_checkpoint
        else:
            training_metadata = train_agents(
                episodes=episodes,
                dataset_paths=train_dataset_paths,
                validation_dataset_paths=validation_dataset_paths or None,
                validation_every=validation_every,
                reward_mode=reward_mode,
                gamma=gamma,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                learning_rate=learning_rate,
                train_start=train_start,
                output_dir=agent_output_dir,
                seed=training_seed,
                return_metadata=True,
            )
            if not isinstance(training_metadata, dict):
                training_metadata = {
                    "best_training_score": training_metadata,
                    "best_validation_tardiness": None,
                    "checkpoint_selection_metric": "training_reward",
                    "reward_mode": reward_mode,
                    "gamma": gamma,
                    "epsilon_decay": epsilon_decay,
                    "epsilon_min": epsilon_min,
                    "learning_rate": learning_rate,
                    "train_start": train_start,
                    "checkpoint_path": str(agent_output_dir / checkpoint_name),
                    "training_history_path": str(agent_output_dir / "training_history.csv"),
                }
            checkpoint_path = Path(str(training_metadata.get("checkpoint_path") or agent_output_dir / checkpoint_name))
            training_row = {
                "training_seed": str(training_seed),
                "episodes": str(episodes),
                "best_training_score": repr(training_metadata.get("best_training_score")),
                "best_validation_tardiness": (
                    ""
                    if training_metadata.get("best_validation_tardiness") is None
                    else repr(training_metadata.get("best_validation_tardiness"))
                ),
                "checkpoint_selection_metric": str(training_metadata.get("checkpoint_selection_metric")),
                "reward_mode": str(training_metadata.get("reward_mode")),
                "gamma": str(training_metadata.get("gamma")),
                "epsilon_decay": str(training_metadata.get("epsilon_decay")),
                "epsilon_min": str(training_metadata.get("epsilon_min")),
                "learning_rate": str(training_metadata.get("learning_rate")),
                "train_start": str(training_metadata.get("train_start")),
                "checkpoint_path": str(checkpoint_path),
                "training_history_path": str(training_metadata.get("training_history_path")),
            }
            training_summary_rows = [
                row for row in training_summary_rows if row["training_seed"] != str(training_seed)
            ]
            training_summary_rows.append(training_row)
            training_summary_by_seed[str(training_seed)] = training_row
            _write_rows(training_summary_path, training_summary_rows)
        for split, instances in (("validation", validation_instances), ("test", test_instances)):
            for instance in instances:
                row_key = (split, instance.instance_id, "DQN", str(training_seed))
                if row_key in completed:
                    continue
                try:
                    result = evaluate_checkpoint(dataset_path=instance.dataset_path, checkpoint_path=checkpoint_path)
                    row = _result_to_row(instance, result)
                except Exception as exc:  # noqa: BLE001 - experiment rows should capture failures.
                    row = _error_row(instance, method="DQN", error=exc)
                row["method"] = "DQN"
                row["split"] = split
                row["training_seed"] = str(training_seed)
                rows.append(row)
                completed.add(row_key)
                _write_rows(csv_path, rows)

    _write_rows(csv_path, rows)
    _write_rows(training_summary_path, training_summary_rows)
    summary_path = output_path / "rl_summary.md"
    summary_path.write_text(make_rl_markdown_summary(rows, training_summary_rows), encoding="utf-8")
    return csv_path, summary_path


PAPER_STUDY_VARIANTS: dict[str, dict[str, object]] = {
    "dense": {
        "reward_mode": "dense_tardiness",
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "learning_rate": 0.001,
        "train_start": 500,
    },
    "sharp": {
        "reward_mode": "sharp",
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "learning_rate": 0.001,
        "train_start": 500,
    },
    "dense_slow_epsilon": {
        "reward_mode": "dense_tardiness",
        "gamma": 0.99,
        "epsilon_decay": 0.997,
        "epsilon_min": 0.01,
        "learning_rate": 0.001,
        "train_start": 500,
    },
    "dense_low_lr": {
        "reward_mode": "dense_tardiness",
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "learning_rate": 0.0005,
        "train_start": 500,
    },
}


def run_paper_study(
    *,
    output_dir: str | Path,
    variants: list[str],
    jobs_values: list[int],
    ddt_values: list[float],
    arrival_rates: list[int],
    train_instance_seeds: list[int],
    validation_instance_seeds: list[int],
    test_instance_seeds: list[int],
    training_seeds: list[int],
    episodes: int = 1000,
    validation_every: int = 50,
    work_centers: int = 5,
    machines_per_work_center: int = 3,
    initial_job_fraction: float = 0.5,
    min_operations: int = 6,
    max_operations: int = 10,
    min_processing_time: int = 60,
    max_processing_time: int = 120,
) -> tuple[Path, Path]:
    """Run a resumable, multi-variant DQN paper study."""

    unknown_variants = [variant for variant in variants if variant not in PAPER_STUDY_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown paper-study variants: {', '.join(unknown_variants)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_path / "paper_study_config.json",
        {
            "study_type": "paper_study",
            "variants": variants,
            "jobs_values": jobs_values,
            "ddt_values": ddt_values,
            "arrival_rates": arrival_rates,
            "train_instance_seeds": train_instance_seeds,
            "validation_instance_seeds": validation_instance_seeds,
            "test_instance_seeds": test_instance_seeds,
            "training_seeds": training_seeds,
            "episodes": episodes,
            "validation_every": validation_every,
            "work_centers": work_centers,
            "machines_per_work_center": machines_per_work_center,
            "initial_job_fraction": initial_job_fraction,
            "min_operations": min_operations,
            "max_operations": max_operations,
            "min_processing_time": min_processing_time,
            "max_processing_time": max_processing_time,
            "variant_parameters": {variant: PAPER_STUDY_VARIANTS[variant] for variant in variants},
        },
    )

    summary_csv_path = output_path / "paper_study_summary.csv"
    summary_rows: list[dict[str, str]] = _read_rows(summary_csv_path)
    for variant in variants:
        variant_output_dir = output_path / variant
        variant_parameters = PAPER_STUDY_VARIANTS[variant]
        csv_path, summary_path = run_rl_generalization_study(
            output_dir=variant_output_dir,
            jobs_values=jobs_values,
            ddt_values=ddt_values,
            arrival_rates=arrival_rates,
            train_instance_seeds=train_instance_seeds,
            validation_instance_seeds=validation_instance_seeds,
            test_instance_seeds=test_instance_seeds,
            training_seeds=training_seeds,
            episodes=episodes,
            validation_every=validation_every,
            reward_mode=str(variant_parameters["reward_mode"]),
            gamma=float(variant_parameters["gamma"]),
            epsilon_decay=float(variant_parameters["epsilon_decay"]),
            epsilon_min=float(variant_parameters["epsilon_min"]),
            learning_rate=float(variant_parameters["learning_rate"]),
            train_start=int(variant_parameters["train_start"]),
            work_centers=work_centers,
            machines_per_work_center=machines_per_work_center,
            initial_job_fraction=initial_job_fraction,
            min_operations=min_operations,
            max_operations=max_operations,
            min_processing_time=min_processing_time,
            max_processing_time=max_processing_time,
        )
        rows = _read_rows(csv_path)
        training_rows = _read_rows(variant_output_dir / "training_summary.csv")
        spt_comparison = _comparison_stats(rows, method="DQN", baseline="SPT_DR_O")
        atc_comparison = _comparison_stats(rows, method="DQN", baseline="ATC_DR_O")
        ok_rows = sum(1 for row in rows if row.get("status", "ok") == "ok")
        error_rows = len(rows) - ok_rows
        summary_rows = [row for row in summary_rows if row["variant"] != variant]
        summary_rows.append(
            {
                "variant": variant,
                "episodes": str(episodes),
                "training_runs": str(len(training_rows)),
                "test_rows": str(sum(1 for row in rows if row.get("split", "test") == "test")),
                "ok_rows": str(ok_rows),
                "error_rows": str(error_rows),
                "dqn_mean_tardiness": _format_float(_method_metric(rows, method="DQN")),
                "spt_mean_tardiness": _format_float(_method_metric(rows, method="SPT_DR_O")),
                "atc_mean_tardiness": _format_float(_method_metric(rows, method="ATC_DR_O")),
                "dqn_minus_spt": _format_float(float(spt_comparison["mean_difference"])),
                "dqn_vs_spt_p": _format_float(float(spt_comparison["wilcoxon_p"])),
                "dqn_vs_spt_wins": str(spt_comparison["wins"]),
                "dqn_vs_spt_losses": str(spt_comparison["losses"]),
                "dqn_vs_spt_ties": str(spt_comparison["ties"]),
                "dqn_minus_atc": _format_float(float(atc_comparison["mean_difference"])),
                "dqn_vs_atc_p": _format_float(float(atc_comparison["wilcoxon_p"])),
                "results_csv": str(csv_path),
                "summary_markdown": str(summary_path),
            }
        )
        _write_rows(summary_csv_path, summary_rows)
        (output_path / "paper_study_summary.md").write_text(
            make_paper_study_summary(summary_rows),
            encoding="utf-8",
        )

    summary_rows.sort(key=lambda row: float(row["dqn_mean_tardiness"]))
    _write_rows(summary_csv_path, summary_rows)
    summary_path = output_path / "paper_study_summary.md"
    summary_path.write_text(make_paper_study_summary(summary_rows), encoding="utf-8")
    return summary_csv_path, summary_path


def make_paper_study_summary(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Paper Study Summary",
        "",
        "Lower tardiness is better. Negative DQN-minus-baseline values mean DQN outperformed that baseline.",
        "",
        "| Rank | Variant | Training runs | DQN mean tardiness | SPT mean tardiness | DQN-SPT | SPT p | W/L/T vs SPT | ATC mean tardiness | DQN-ATC | ATC p | Errors |",
        "|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(sorted(rows, key=lambda item: float(item["dqn_mean_tardiness"])), 1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {}/{}/{} | {} | {} | {} | {} |".format(
                rank,
                row["variant"],
                row["training_runs"],
                row["dqn_mean_tardiness"],
                row["spt_mean_tardiness"],
                row["dqn_minus_spt"],
                row["dqn_vs_spt_p"],
                row["dqn_vs_spt_wins"],
                row["dqn_vs_spt_losses"],
                row["dqn_vs_spt_ties"],
                row["atc_mean_tardiness"],
                row["dqn_minus_atc"],
                row["dqn_vs_atc_p"],
                row["error_rows"],
            )
        )

    lines.extend(
        [
            "",
            "## Variant Artifacts",
            "",
            "| Variant | Results CSV | Summary |",
            "|---|---|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: item["variant"]):
        lines.append(f"| {row['variant']} | {row['results_csv']} | {row['summary_markdown']} |")
    return "\n".join(lines) + "\n"


def make_rl_markdown_summary(rows: list[dict[str, str]], training_rows: list[dict[str, str]]) -> str:
    method_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    baseline_by_method_instance: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    split_counts: dict[str, int] = defaultdict(int)
    error_rows = [row for row in rows if row.get("status", "ok") != "ok"]
    for row in rows:
        split_counts[row.get("split", "test")] += 1
        if row.get("split", "test") != "test" or row.get("status", "ok") != "ok":
            continue
        method_rows[row["method"]].append(row)
        if row["method"] != "DQN":
            baseline_by_method_instance[row["method"]][row["instance_id"]] = row

    summary_rows = []
    for method, values in method_rows.items():
        tardiness_values = [float(row["tardiness_rate"]) for row in values]
        summary_rows.append(
            {
                "method": method,
                "n": len(values),
                "mean_tardiness": _mean(tardiness_values),
                "std_tardiness": _std(tardiness_values),
                "ci95_tardiness": _ci95(tardiness_values),
                "mean_makespan": _mean([float(row["makespan"]) for row in values]),
                "mean_utilization": _mean([float(row["mean_machine_utilization"]) for row in values]),
            }
        )
    summary_rows.sort(key=lambda row: row["mean_tardiness"])

    lines = [
        "# RL Generalization Study Summary",
        "",
        f"Training runs: {len(training_rows)}",
        f"Test result rows: {sum(len(values) for values in method_rows.values())}",
        f"Failed result rows: {len(error_rows)}",
        "",
        "## Rows by Split",
        "",
        "| Split | Rows |",
        "|---|---:|",
    ]
    for split, count in sorted(split_counts.items()):
        lines.append(f"| {split} | {count} |")

    lines.extend(
        [
            "",
            "## Training Runs",
            "",
            "| Training seed | Episodes | Best training score | Best validation tardiness | Selection metric | Checkpoint |",
            "|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in training_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["training_seed"],
                row["episodes"],
                row["best_training_score"],
                row.get("best_validation_tardiness", ""),
                row.get("checkpoint_selection_metric", "training_reward"),
                row["checkpoint_path"],
            )
        )

    dqn_by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in method_rows.get("DQN", []):
        if row.get("training_seed"):
            dqn_by_seed[row["training_seed"]].append(row)
    if dqn_by_seed:
        lines.extend(
            [
                "",
                "## DQN by Training Seed",
                "",
                "| Training seed | Held-out instances | Mean tardiness | Std | 95% CI |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for training_seed, seed_rows in sorted(dqn_by_seed.items(), key=lambda item: int(item[0])):
            tardiness_values = [float(row["tardiness_rate"]) for row in seed_rows]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    training_seed,
                    len(seed_rows),
                    _format_float(_mean(tardiness_values)),
                    _format_float(_std(tardiness_values)),
                    _format_float(_ci95(tardiness_values)),
                )
            )

    lines.extend(
        [
            "",
            "## Test Ranking",
            "",
            "| Rank | Method | n | Mean tardiness | Std | 95% CI | Mean makespan | Mean utilization |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary_rows, 1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {}% |".format(
                rank,
                row["method"],
                row["n"],
                _format_float(row["mean_tardiness"]),
                _format_float(row["std_tardiness"]),
                _format_float(row["ci95_tardiness"]),
                _format_float(row["mean_makespan"]),
                _format_float(row["mean_utilization"]),
            )
        )

    if "DQN" in method_rows and baseline_by_method_instance:
        comparison_rows = []
        for baseline_method, baseline_by_instance in baseline_by_method_instance.items():
            differences = []
            wins = losses = ties = 0
            for row in method_rows["DQN"]:
                baseline = baseline_by_instance.get(row["instance_id"])
                if baseline is None:
                    continue
                difference = float(row["tardiness_rate"]) - float(baseline["tardiness_rate"])
                differences.append(difference)
                if difference < 0:
                    wins += 1
                elif difference > 0:
                    losses += 1
                else:
                    ties += 1
            comparison_rows.append((baseline_method, differences, wins, losses, ties))
        comparison_rows.sort(key=lambda item: _mean(item[1]) if item[1] else 0.0)
        lines.extend(
            [
                "",
                "## DQN Against Baselines",
                "",
                "Negative differences mean DQN had lower tardiness than the baseline.",
                "",
                "| Baseline | Compared pairs | Mean tardiness difference | Wilcoxon p | Wins | Losses | Ties |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for baseline_method, differences, wins, losses, ties in comparison_rows:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    baseline_method,
                    len(differences),
                    _format_float(_mean(differences)),
                    _format_float(_wilcoxon_p_value(differences)),
                    wins,
                    losses,
                    ties,
                )
            )

    lines.extend(
        [
        "",
        "## Interpretation Guardrail",
        "",
        "Only held-out test rows are used for the ranking and paired comparisons. Validation rows are available in the raw CSV for checkpoint-selection auditing.",
        ]
    )

    return "\n".join(lines) + "\n"


def make_markdown_summary(rows: list[dict[str, str]], *, reference_method: str = "SPT_DR_O") -> str:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_instance: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    error_rows = [row for row in rows if row.get("status", "ok") != "ok"]
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        by_method[row["method"]].append(row)
        by_instance[row["instance_id"]][row["method"]] = row

    summary_rows = []
    for method, method_rows in by_method.items():
        tardiness_values = [float(row["tardiness_rate"]) for row in method_rows]
        makespans = [float(row["makespan"]) for row in method_rows]
        utilizations = [float(row["mean_machine_utilization"]) for row in method_rows]
        summary_rows.append(
            {
                "method": method,
                "n": len(method_rows),
                "mean_tardiness": _mean(tardiness_values),
                "std_tardiness": _std(tardiness_values),
                "ci95_tardiness": _ci95(tardiness_values),
                "mean_makespan": _mean(makespans),
                "mean_utilization": _mean(utilizations),
            }
        )
    summary_rows.sort(key=lambda row: row["mean_tardiness"])

    lines = [
        "# Experiment Matrix Summary",
        "",
        f"Evaluated {len(by_instance)} generated instances and {len(by_method)} methods.",
        f"Failed result rows: {len(error_rows)}.",
        "",
        "Lower tardiness is better. The confidence interval is a normal-approximation 95% CI across evaluated instances.",
        "",
        "## Method Ranking",
        "",
        "| Rank | Method | n | Mean tardiness | Std | 95% CI | Mean makespan | Mean utilization |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(summary_rows, 1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {}% |".format(
                rank,
                row["method"],
                row["n"],
                _format_float(row["mean_tardiness"]),
                _format_float(row["std_tardiness"]),
                _format_float(row["ci95_tardiness"]),
                _format_float(row["mean_makespan"]),
                _format_float(row["mean_utilization"]),
            )
        )

    if reference_method in by_method:
        lines.extend(
            [
                "",
                f"## Paired Comparison Against `{reference_method}`",
                "",
                "| Method | Compared instances | Mean tardiness difference | Wilcoxon p | Wins | Losses | Ties |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        paired_rows = []
        for method in by_method:
            if method == reference_method:
                continue
            differences = []
            wins = losses = ties = 0
            for instance_methods in by_instance.values():
                if reference_method not in instance_methods or method not in instance_methods:
                    continue
                method_value = float(instance_methods[method]["tardiness_rate"])
                reference_value = float(instance_methods[reference_method]["tardiness_rate"])
                difference = method_value - reference_value
                differences.append(difference)
                if difference < 0:
                    wins += 1
                elif difference > 0:
                    losses += 1
                else:
                    ties += 1
            paired_rows.append((method, differences, wins, losses, ties))
        paired_rows.sort(key=lambda item: _mean(item[1]) if item[1] else 0.0)
        for method, differences, wins, losses, ties in paired_rows:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    method,
                    len(differences),
                    _format_float(_mean(differences)),
                    _format_float(_wilcoxon_p_value(differences)),
                    wins,
                    losses,
                    ties,
                )
            )

    lines.extend(
        [
            "",
            "## Generated Conditions",
            "",
            "| Jobs | Machines | DDT | Arrival rate | Seeds |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    conditions: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        conditions[(row["jobs"], row["machines"], row["ddt"], row["arrival_rate"])].add(row["seed"])
    for (jobs, machines, ddt, arrival_rate), condition_seeds in sorted(conditions.items()):
        lines.append(f"| {jobs} | {machines} | {ddt} | {arrival_rate} | {', '.join(sorted(condition_seeds))} |")

    return "\n".join(lines) + "\n"
