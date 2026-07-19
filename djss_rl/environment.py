"""Environment construction for the DJSS RL project."""

from __future__ import annotations

from pathlib import Path

from . import core

DEFAULT_DATASET = "Dataset 50_0.5_0.02.ini"

Operation = core.Operation
Job = core.Job
Work_Center = core.Work_Center
Machine = core.Machine
World = core.World
BaseScenario = core.BaseScenario
Scenario = core.Scenario
Job_Shop_Env = core.Job_Shop_Env
OperationSelector = core.OperationSelector


def set_runtime_options(
    *,
    maintenance_integrated: bool = False,
    validation: bool = False,
    reward_mode: str = "sharp",
) -> None:
    """Set notebook-compatible runtime flags used by the extracted classes."""

    core.maintenance_integrated = maintenance_integrated
    core.validation = validation
    core.reward_mode = reward_mode


def make_env(
    dataset_path: str | Path | None = DEFAULT_DATASET,
    *,
    maintenance_integrated: bool = False,
    validation: bool = False,
    reward_mode: str = "sharp",
) -> Job_Shop_Env:
    """Create the job-shop environment from the restored dataset by default."""

    set_runtime_options(
        maintenance_integrated=maintenance_integrated,
        validation=validation,
        reward_mode=reward_mode,
    )
    return core.make_env(dataset_path=str(dataset_path) if dataset_path is not None else None)
