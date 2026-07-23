"""Generate the paper figures from committed experiment CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
REPO_ROOT = PAPER_DIR.parents[1]
FIGURE_DIR = PAPER_DIR / "figures"

GENERATED_RESULTS = (
    REPO_ROOT
    / "outputs"
    / "expanded-dense-multiseed-20260721"
    / "dense"
    / "rl_results.csv"
)
BENCHMARK_RESULTS = (
    REPO_ROOT
    / "outputs"
    / "or-library-benchmark-study-20260723-clean"
    / "benchmark_results.csv"
)
GENERATED_TRACE = (
    REPO_ROOT
    / "outputs"
    / "expanded-dense-multiseed-20260721"
    / "dense"
    / "policy-trace"
    / "checkpoint_policy_trace.csv"
)
BENCHMARK_TRACE = (
    REPO_ROOT
    / "outputs"
    / "or-library-benchmark-study-20260723-clean"
    / "policy-trace"
    / "checkpoint_policy_trace.csv"
)

METHODS = ["SPT_DR_O", "DQN", "ATC_DR_O", "MRT_DR_O"]
METHOD_LABELS = ["SPT", "DQN", "ATC", "MRT"]
COLORS = ["#2878B5", "#D95319", "#3A923A", "#7A5195"]
TRACE_ACTIONS = [
    "MRT_DR_O",
    "LRT_DR_O",
    "SPT_DR_O",
    "EDD_DR_O",
    "MCR_DR_O",
    "CR_DR_O",
    "ATC_DR_O",
    "LSPO_DR_O",
    "SLK_DR_O",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_tardiness(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in read_rows(path):
        if row.get("split", "test") != "test" or row.get("status", "ok") != "ok":
            continue
        if row["method"] in METHODS:
            values[row["method"]].append(float(row["tardiness_rate"]))
    return {method: float(np.mean(values[method])) for method in METHODS}


def make_performance_figure() -> None:
    generated = mean_tardiness(GENERATED_RESULTS)
    benchmark = mean_tardiness(BENCHMARK_RESULTS)
    studies = [generated, benchmark]

    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    x = np.arange(2)
    width = 0.18
    for index, (method, label, color) in enumerate(zip(METHODS, METHOD_LABELS, COLORS)):
        offset = (index - 1.5) * width
        bars = ax.bar(
            x + offset,
            [study[method] for study in studies],
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)

    ax.set_xticks(x, ["Generated test matrix", "OR-Library-derived"])
    ax.set_ylabel("Mean terminal tardiness rate")
    ax.set_ylim(0, 0.64)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.7)
    _save(fig, "performance_comparison")


def trace_summary(path: Path) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for row in read_rows(path):
        if row.get("status", "ok") != "ok":
            continue
        seed = row["checkpoint_label"]
        for method in TRACE_ACTIONS:
            count = int(float(row.get(f"count_{method}", "0") or 0))
            counts[seed][method] += count
            totals[seed] += count
    return {
        seed: {
            action: counts[seed][action] / totals[seed] if totals[seed] else 0.0
            for action in TRACE_ACTIONS
        }
        for seed in sorted(counts, key=int)
    }


def make_policy_trace_figure() -> None:
    generated = trace_summary(GENERATED_TRACE)
    benchmark = trace_summary(BENCHMARK_TRACE)
    seeds = ["11", "22", "33"]
    actions = ["SPT_DR_O", "MRT_DR_O", "ATC_DR_O"]
    labels = ["SPT", "MRT", "ATC"]

    matrix = []
    row_labels = []
    for study_name, summary in (("Generated", generated), ("Benchmark-derived", benchmark)):
        for seed in seeds:
            matrix.append([summary[seed].get(action, 0.0) for action in actions])
            row_labels.append(f"{study_name}, seed {seed}")

    array = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    image = ax.imshow(array, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_xlabel("Selected dispatching-rule action")

    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = array[row, column]
            text_color = "white" if value > 0.6 else "black"
            ax.text(
                column,
                row,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold" if value == 1 else "normal",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Decision fraction")
    fig.tight_layout(pad=0.7)
    _save(fig, "policy_collapse")


def generated_regime_gap(field: str) -> tuple[list[str], list[float]]:
    rows = [
        row
        for row in read_rows(GENERATED_RESULTS)
        if row.get("split") == "test" and row.get("status") == "ok"
    ]
    baseline_by_instance = {
        row["instance_id"]: float(row["tardiness_rate"])
        for row in rows
        if row["method"] == "SPT_DR_O"
    }
    differences: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "DQN":
            continue
        differences[row[field]].append(
            float(row["tardiness_rate"]) - baseline_by_instance[row["instance_id"]]
        )
    labels = sorted(differences, key=float)
    return labels, [float(np.mean(differences[label])) for label in labels]


def make_regime_figure() -> None:
    fields = [
        ("jobs", "Jobs"),
        ("ddt", "Due-date tightness"),
        ("arrival_rate", "Mean interarrival parameter"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7))
    for ax, (field, title) in zip(axes, fields):
        labels, values = generated_regime_gap(field)
        colors = ["#D95319" if value >= 0 else "#2878B5" for value in values]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.bar_label(bars, labels=[f"{value:+.3f}" for value in values], padding=2, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.004, 0.031)
    axes[0].set_ylabel("Mean DQN - SPT tardiness")
    fig.tight_layout(pad=0.65, w_pad=0.8)
    _save(fig, "generated_regime_gap")


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    make_performance_figure()
    make_policy_trace_figure()
    make_regime_figure()
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
