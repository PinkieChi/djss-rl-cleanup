"""Safe execution helpers for the cleaned DJSS RL notebook.

The project still keeps most research code in the notebook. This module gives
the notebook a repeatable terminal interface while the code is being refactored
into regular Python modules.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_NOTEBOOK = "DQN_based_Dynamic_Job_Shop_Scheduling_tardiness.ipynb"
DEFAULT_DATASET = "Dataset 50_0.5_0.02.ini"


@dataclass
class NotebookRunSummary:
    executed_cells: list[int]
    skipped_cells: list[tuple[int, str]]
    jobs: int | None
    machines: int | None
    operations: int | None
    total_timestamp: int | float | None
    observation_shape: tuple[int, ...] | None
    action_space: int | None
    costs: list[float]


def _set_env_flag(name: str, enabled: bool) -> None:
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)


def execute_notebook(
    project_dir: Path | str = Path("."),
    *,
    run_evaluation: bool = False,
    run_training: bool = False,
    run_wandb_smoke_test: bool = False,
    notebook_name: str = DEFAULT_NOTEBOOK,
    dataset_path: str = DEFAULT_DATASET,
) -> NotebookRunSummary:
    """Execute notebook code cells with explicit safety guards.

    By default this performs the safe smoke run: no training, no evaluation,
    no W&B online sync, and no notebook outputs are written back to disk.
    """

    project_path = Path(project_dir).resolve()
    notebook_path = project_path / notebook_name
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "djss-rl-matplotlib"))
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ["DJSS_DATASET_PATH"] = dataset_path
    _set_env_flag("RUN_DJSS_EVALUATION", run_evaluation)
    _set_env_flag("RUN_DJSS_TRAINING", run_training)
    _set_env_flag("RUN_WANDB_SMOKE_TEST", run_wandb_smoke_test)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__main__"}
    executed_cells: list[int] = []
    skipped_cells: list[tuple[int, str]] = []

    previous_cwd = Path.cwd()
    os.chdir(project_path)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            stripped = source.lstrip()
            if not stripped:
                skipped_cells.append((index, "empty"))
                continue
            if stripped.startswith("%%") or stripped.startswith("!"):
                skipped_cells.append((index, "magic/shell"))
                continue
            print(f">>> executing cell {index}", flush=True)
            exec(compile(source, f"cell_{index}", "exec"), namespace)
            executed_cells.append(index)
    finally:
        os.chdir(previous_cwd)

    world = namespace.get("world")
    env = namespace.get("env")
    observation_shape = None
    action_space = None
    if env is not None:
        observation_shape = tuple(env.observation_space.shape)
        action_space = int(env.action_space.n)

    return NotebookRunSummary(
        executed_cells=executed_cells,
        skipped_cells=skipped_cells,
        jobs=len(world.jobs) if world is not None else None,
        machines=len(world.machines) if world is not None else None,
        operations=getattr(world, "operations", None) if world is not None else None,
        total_timestamp=getattr(world, "total_timestamp", None) if world is not None else None,
        observation_shape=observation_shape,
        action_space=action_space,
        costs=[float(value) for value in namespace.get("costs", [])],
    )
