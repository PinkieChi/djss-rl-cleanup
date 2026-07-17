"""Evaluation helpers for baseline dispatching rules and saved agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agent import DQNAgent
from .environment import DEFAULT_DATASET, make_env

DEFAULT_CHECKPOINT = (
    "Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth"
)
DEFAULT_HIDDEN_LAYERS = 7
DEFAULT_NEURONS_PER_LAYER = [207, 145, 78, 79, 205, 105, 217]
DEFAULT_BATCH_SIZE = 32
HEURISTICS = [
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


@dataclass(frozen=True)
class SchedulingResult:
    name: str
    tardiness_rate: float
    makespan: int | float
    corrective_maintenance_actions: int | float
    preventive_maintenance_actions: int | float
    energy_consumption: float
    mean_machine_utilization: float


def run_scheduling(env, world, *, name: str, decision_rule: int | None = None, agent: DQNAgent | None = None) -> SchedulingResult:
    """Run a full scheduling episode with either a dispatching rule or an agent."""

    env.reset()

    while not all(job.CRJ == 1 for job in world.jobs):
        if not (any(job.legal for job in world.jobs) and any(machine.request for machine in world.machines)):
            env.update_state()
            continue

        world.ready_machine = next((machine for machine in world.machines if machine.request), None)
        env.get_legal_actions(world.ready_machine)

        if world.ready_machine.legal_actions:
            if agent is not None:
                state = env._get_obs
                action = agent.get_action(state, world)
                env.step(action)
            elif decision_rule is not None:
                env.step(decision_rule)
            else:
                raise ValueError("Either decision_rule or agent must be provided.")
        else:
            world.ready_machine.request = False

    return SchedulingResult(
        name=name,
        tardiness_rate=float(world.tardiness_rate),
        makespan=env._get_makespan,
        corrective_maintenance_actions=float(np.sum([machine.CM_actions_counts for machine in world.machines])),
        preventive_maintenance_actions=float(np.sum([machine.PM_actions_counts for machine in world.machines])),
        energy_consumption=float(world.total_energy_consumption),
        mean_machine_utilization=float(np.mean([machine.utilization for machine in world.machines]) * 100),
    )


def evaluate_baselines(dataset_path: str | Path | None = DEFAULT_DATASET) -> list[SchedulingResult]:
    """Evaluate all dispatching-rule baselines."""

    env = make_env(dataset_path=dataset_path)
    world = env.world
    return [
        run_scheduling(env, world, name=HEURISTICS[index], decision_rule=index)
        for index, _ in enumerate(env.action_mapping_operation_sequencing_list)
    ]


def evaluate_checkpoint(
    dataset_path: str | Path | None = DEFAULT_DATASET,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
    neurons_per_layer: list[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SchedulingResult:
    """Evaluate a saved DQN checkpoint."""

    env = make_env(dataset_path=dataset_path)
    world = env.world
    neurons = neurons_per_layer or DEFAULT_NEURONS_PER_LAYER
    agent = DQNAgent(
        input_dim=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons,
        batch_size=batch_size,
    )
    agent.epsilon = 0
    agent.load_model(str(checkpoint_path))
    return run_scheduling(env, world, name="Ours", agent=agent)


def evaluate_all(
    dataset_path: str | Path | None = DEFAULT_DATASET,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
) -> list[SchedulingResult]:
    """Evaluate baselines and the saved checkpoint."""

    return evaluate_baselines(dataset_path=dataset_path) + [
        evaluate_checkpoint(dataset_path=dataset_path, checkpoint_path=checkpoint_path)
    ]

