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
    load_checkpoint_agent,
    run_scheduling,
    trace_scheduling_actions,
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


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] + fraction * (sorted_values[upper_index] - sorted_values[lower_index])


def _iqr(values: list[float]) -> float:
    return _percentile(values, 0.75) - _percentile(values, 0.25)


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * _std(values) / (len(values) ** 0.5)


def _wilcoxon_p_value(values: list[float]) -> float:
    nonzero_values = [value for value in values if value != 0]
    if len(nonzero_values) < 2:
        return float("nan")
    return float(wilcoxon(nonzero_values).pvalue)


def _paired_rank_biserial(values: list[float]) -> float:
    """Return paired rank-biserial effect size for signed differences."""

    nonzero_values = [value for value in values if value != 0]
    if not nonzero_values:
        return 0.0
    sorted_pairs = sorted((abs(value), value) for value in nonzero_values)
    signed_rank_sum = 0.0
    total_rank_sum = 0.0
    index = 0
    while index < len(sorted_pairs):
        next_index = index + 1
        while next_index < len(sorted_pairs) and sorted_pairs[next_index][0] == sorted_pairs[index][0]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2
        for _, value in sorted_pairs[index:next_index]:
            signed_rank_sum += average_rank if value > 0 else -average_rank
            total_rank_sum += average_rank
        index = next_index
    return signed_rank_sum / total_rank_sum if total_rank_sum else 0.0


def _label_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


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
        "median_difference": _median(differences),
        "std_difference": _std(differences),
        "wilcoxon_p": _wilcoxon_p_value(differences),
        "rank_biserial": _paired_rank_biserial(differences),
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


def _summarize_method_rows(method: str, values: list[dict[str, str]]) -> dict[str, object]:
    tardiness_values = [float(row["tardiness_rate"]) for row in values]
    return {
        "method": method,
        "n": len(values),
        "mean_tardiness": _mean(tardiness_values),
        "median_tardiness": _median(tardiness_values),
        "std_tardiness": _std(tardiness_values),
        "iqr_tardiness": _iqr(tardiness_values),
        "ci95_tardiness": _ci95(tardiness_values),
        "mean_makespan": _mean([float(row["makespan"]) for row in values]),
        "mean_utilization": _mean([float(row["mean_machine_utilization"]) for row in values]),
    }


def _append_regime_breakdown(lines: list[str], rows: list[dict[str, str]], *, methods: list[str]) -> None:
    dimensions = [
        ("jobs", "Jobs"),
        ("ddt", "DDT"),
        ("arrival_rate", "Arrival rate"),
    ]
    for field, label in dimensions:
        regime_values = sorted({row[field] for row in rows}, key=_label_sort_key)
        if not regime_values:
            continue
        lines.extend(
            [
                "",
                f"### By {label}",
                "",
                f"| {label} | Method | n | Mean tardiness | Median | 95% CI |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for regime_value in regime_values:
            for method in methods:
                method_values = [
                    row
                    for row in rows
                    if row[field] == regime_value and row["method"] == method and row.get("status", "ok") == "ok"
                ]
                if not method_values:
                    continue
                tardiness_values = [float(row["tardiness_rate"]) for row in method_values]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} |".format(
                        regime_value,
                        method,
                        len(method_values),
                        _format_float(_mean(tardiness_values)),
                        _format_float(_median(tardiness_values)),
                        _format_float(_ci95(tardiness_values)),
                    )
                )


def _checkpoint_label(checkpoint_path: Path) -> str:
    for part in reversed(checkpoint_path.parts):
        if part.startswith("seed-"):
            return part.removeprefix("seed-")
    return checkpoint_path.stem


def run_checkpoint_generalization_study(
    *,
    output_dir: str | Path,
    checkpoint_paths: list[str | Path],
    jobs_values: list[int],
    ddt_values: list[float],
    arrival_rates: list[int],
    test_instance_seeds: list[int],
    checkpoint_labels: list[str] | None = None,
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
    """Evaluate existing DQN checkpoints on a new held-out generated-instance matrix."""

    checkpoints = [Path(path) for path in checkpoint_paths]
    if not checkpoints:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    if checkpoint_labels is not None and len(checkpoint_labels) != len(checkpoints):
        raise ValueError("checkpoint_labels must match checkpoint_paths length")
    labels = checkpoint_labels or [_checkpoint_label(path) for path in checkpoints]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_path / "checkpoint_study_config.json",
        {
            "study_type": "checkpoint_generalization",
            "checkpoint_paths": [str(path) for path in checkpoints],
            "checkpoint_labels": labels,
            "jobs_values": jobs_values,
            "ddt_values": ddt_values,
            "arrival_rates": arrival_rates,
            "test_instance_seeds": test_instance_seeds,
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

    csv_path = output_path / "checkpoint_results.csv"
    rows: list[dict[str, str]] = _read_rows(csv_path)
    completed = {_experiment_row_key(row) for row in rows}
    for instance in test_instances:
        for index, heuristic in enumerate(HEURISTICS):
            row_key = ("test", instance.instance_id, heuristic, "")
            if row_key in completed:
                continue
            try:
                env = make_env(dataset_path=instance.dataset_path)
                result = run_scheduling(env, env.world, name=heuristic, decision_rule=index)
                row = _result_to_row(instance, result)
            except Exception as exc:  # noqa: BLE001 - experiment rows should capture failures.
                row = _error_row(instance, method=heuristic, error=exc)
            row["split"] = "test"
            row["training_seed"] = ""
            row["checkpoint_path"] = ""
            rows.append(row)
            completed.add(row_key)
            _write_rows(csv_path, rows)

    for checkpoint, label in zip(checkpoints, labels):
        sample_env = make_env(dataset_path=test_instances[0].dataset_path)
        agent = load_checkpoint_agent(
            checkpoint_path=checkpoint,
            input_dim=sample_env.observation_space.shape[0],
            action_size=sample_env.action_space.n,
        )
        for instance in test_instances:
            row_key = ("test", instance.instance_id, "DQN", label)
            if row_key in completed:
                continue
            try:
                env = make_env(dataset_path=instance.dataset_path)
                result = run_scheduling(env, env.world, name="Ours", agent=agent)
                row = _result_to_row(instance, result)
            except Exception as exc:  # noqa: BLE001 - experiment rows should capture failures.
                row = _error_row(instance, method="DQN", error=exc)
            row["method"] = "DQN"
            row["split"] = "test"
            row["training_seed"] = label
            row["checkpoint_path"] = str(checkpoint)
            rows.append(row)
            completed.add(row_key)
            _write_rows(csv_path, rows)

    training_rows = [
        {
            "training_seed": label,
            "episodes": "pretrained",
            "best_training_score": "",
            "best_validation_tardiness": "",
            "checkpoint_selection_metric": "pretrained_checkpoint",
            "checkpoint_path": str(checkpoint),
        }
        for checkpoint, label in zip(checkpoints, labels)
    ]
    summary_path = output_path / "checkpoint_summary.md"
    summary = make_rl_markdown_summary(rows, training_rows)
    summary += (
        "\n## Publication Use\n\n"
        "This study evaluates already-trained checkpoints on additional held-out generated instances. "
        "Use it as an external generalization check; do not treat it as a replacement for retraining on a larger protocol.\n"
    )
    summary_path.write_text(summary, encoding="utf-8")
    return csv_path, summary_path


def _policy_trace_row(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    checkpoint_label: str,
    trace,
) -> dict[str, str]:
    row = {
        "instance_id": dataset_path.stem,
        "dataset_path": str(dataset_path),
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "tardiness_rate": repr(trace.result.tardiness_rate),
        "makespan": repr(trace.result.makespan),
        "mean_machine_utilization": repr(trace.result.mean_machine_utilization),
        "decisions": str(trace.decisions),
        "dominant_action": trace.dominant_action,
        "dominant_fraction": repr(trace.dominant_fraction),
        "status": "ok",
        "error": "",
    }
    for heuristic in HEURISTICS:
        count = trace.action_counts.get(heuristic, 0)
        row[f"count_{heuristic}"] = str(count)
        row[f"fraction_{heuristic}"] = repr(count / trace.decisions if trace.decisions else 0.0)
    return row


def _policy_trace_error_row(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    checkpoint_label: str,
    error: Exception,
) -> dict[str, str]:
    row = {
        "instance_id": dataset_path.stem,
        "dataset_path": str(dataset_path),
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "tardiness_rate": "nan",
        "makespan": "nan",
        "mean_machine_utilization": "nan",
        "decisions": "0",
        "dominant_action": "",
        "dominant_fraction": "0.0",
        "status": "error",
        "error": str(error),
    }
    for heuristic in HEURISTICS:
        row[f"count_{heuristic}"] = "0"
        row[f"fraction_{heuristic}"] = "0.0"
    return row


def make_policy_trace_summary(rows: list[dict[str, str]], *, checkpoint_path: str | Path) -> str:
    """Create a Markdown summary for DQN action-selection diagnostics."""

    ok_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    total_decisions = sum(int(row["decisions"]) for row in ok_rows)
    lines = [
        "# Policy Trace Summary",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Instances traced: {len(ok_rows)}",
        f"- Total dispatching decisions: {total_decisions}",
    ]
    if not ok_rows:
        lines.extend(["", "No successful policy traces were completed."])
        return "\n".join(lines) + "\n"

    tardiness_values = [float(row["tardiness_rate"]) for row in ok_rows]
    lines.extend(
        [
            f"- Mean tardiness: {_format_float(_mean(tardiness_values))}",
            "",
            "## Action Distribution",
            "",
            "| Action | Count | Fraction |",
            "|---|---:|---:|",
        ]
    )
    for heuristic in HEURISTICS:
        count = sum(int(row[f"count_{heuristic}"]) for row in ok_rows)
        fraction = count / total_decisions if total_decisions else 0.0
        lines.append(f"| {heuristic} | {count} | {_format_float(fraction)} |")

    dominant_counts: dict[str, int] = defaultdict(int)
    for row in ok_rows:
        dominant_counts[row["dominant_action"]] += 1
    lines.extend(
        [
            "",
            "## Dominant Action Per Instance",
            "",
            "| Action | Instances |",
            "|---|---:|",
        ]
    )
    for action, count in sorted(dominant_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## Instance Details",
            "",
            "| Instance | Decisions | Dominant action | Dominant fraction | Tardiness |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in ok_rows:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                row["instance_id"],
                row["decisions"],
                row["dominant_action"],
                _format_float(float(row["dominant_fraction"])),
                _format_float(float(row["tardiness_rate"])),
            )
        )

    error_rows = [row for row in rows if row.get("status") == "error"]
    if error_rows:
        lines.extend(["", "## Errors", "", "| Instance | Error |", "|---|---|"])
        for row in error_rows:
            lines.append(f"| {row['instance_id']} | {row['error']} |")
    return "\n".join(lines) + "\n"


def run_policy_trace_study(
    *,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    dataset_paths: list[str | Path],
    checkpoint_label: str = "DQN",
) -> tuple[Path, Path]:
    """Trace which dispatching-rule action a checkpoint selects on datasets."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    datasets = [Path(path) for path in dataset_paths]
    if not datasets:
        raise ValueError("dataset_paths must contain at least one dataset")
    for dataset in datasets:
        if not dataset.exists():
            raise FileNotFoundError(dataset)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_path / "policy_trace_config.json",
        {
            "study_type": "policy_trace",
            "checkpoint_path": str(checkpoint),
            "checkpoint_label": checkpoint_label,
            "dataset_paths": [str(path) for path in datasets],
        },
    )

    sample_env = make_env(dataset_path=datasets[0])
    agent = load_checkpoint_agent(
        checkpoint_path=checkpoint,
        input_dim=sample_env.observation_space.shape[0],
        action_size=sample_env.action_space.n,
    )

    csv_path = output_path / "policy_trace.csv"
    rows: list[dict[str, str]] = []
    for dataset_path in datasets:
        try:
            env = make_env(dataset_path=dataset_path)
            trace = trace_scheduling_actions(env, env.world, name=checkpoint_label, agent=agent)
            row = _policy_trace_row(
                dataset_path=dataset_path,
                checkpoint_path=checkpoint,
                checkpoint_label=checkpoint_label,
                trace=trace,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should capture failures.
            row = _policy_trace_error_row(
                dataset_path=dataset_path,
                checkpoint_path=checkpoint,
                checkpoint_label=checkpoint_label,
                error=exc,
            )
        rows.append(row)
        _write_rows(csv_path, rows)

    summary_path = output_path / "policy_trace_summary.md"
    summary_path.write_text(make_policy_trace_summary(rows, checkpoint_path=checkpoint), encoding="utf-8")
    return csv_path, summary_path


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
        summary_rows.append(_summarize_method_rows(method, values))
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
                "| Training seed | Held-out instances | Mean tardiness | Median | Std | 95% CI |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for training_seed, seed_rows in sorted(dqn_by_seed.items(), key=lambda item: _label_sort_key(item[0])):
            tardiness_values = [float(row["tardiness_rate"]) for row in seed_rows]
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    training_seed,
                    len(seed_rows),
                    _format_float(_mean(tardiness_values)),
                    _format_float(_median(tardiness_values)),
                    _format_float(_std(tardiness_values)),
                    _format_float(_ci95(tardiness_values)),
                )
            )

    lines.extend(
        [
            "",
            "## Test Ranking",
            "",
            "| Rank | Method | n | Mean tardiness | Median | IQR | Std | 95% CI | Mean makespan | Mean utilization |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary_rows, 1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {}% |".format(
                rank,
                row["method"],
                row["n"],
                _format_float(row["mean_tardiness"]),
                _format_float(row["median_tardiness"]),
                _format_float(row["iqr_tardiness"]),
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
                "| Baseline | Compared pairs | Mean difference | Median difference | Wilcoxon p | Rank-biserial r | Wins | Losses | Ties |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for baseline_method, differences, wins, losses, ties in comparison_rows:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    baseline_method,
                    len(differences),
                    _format_float(_mean(differences)),
                    _format_float(_median(differences)),
                    _format_float(_wilcoxon_p_value(differences)),
                    _format_float(_paired_rank_biserial(differences)),
                    wins,
                    losses,
                    ties,
                )
            )

    ok_test_rows = [row for row in rows if row.get("split", "test") == "test" and row.get("status", "ok") == "ok"]
    if ok_test_rows:
        ranked_methods = [str(row["method"]) for row in summary_rows[: min(4, len(summary_rows))]]
        lines.extend(["", "## Regime Breakdown"])
        _append_regime_breakdown(lines, ok_test_rows, methods=ranked_methods)

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
        summary_rows.append(_summarize_method_rows(method, method_rows))
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
        "| Rank | Method | n | Mean tardiness | Median | IQR | Std | 95% CI | Mean makespan | Mean utilization |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(summary_rows, 1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {}% |".format(
                rank,
                row["method"],
                row["n"],
                _format_float(row["mean_tardiness"]),
                _format_float(row["median_tardiness"]),
                _format_float(row["iqr_tardiness"]),
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
                "| Method | Compared instances | Mean difference | Median difference | Wilcoxon p | Rank-biserial r | Wins | Losses | Ties |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    method,
                    len(differences),
                    _format_float(_mean(differences)),
                    _format_float(_median(differences)),
                    _format_float(_wilcoxon_p_value(differences)),
                    _format_float(_paired_rank_biserial(differences)),
                    wins,
                    losses,
                    ties,
                )
            )

    ok_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if ok_rows:
        ranked_methods = [str(row["method"]) for row in summary_rows[: min(4, len(summary_rows))]]
        lines.extend(["", "## Regime Breakdown"])
        _append_regime_breakdown(lines, ok_rows, methods=ranked_methods)

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
