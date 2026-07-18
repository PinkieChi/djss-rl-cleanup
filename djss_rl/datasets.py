"""Dataset generation utilities for repeatable DJSS experiments."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
import random


SCALE_KEY = "scale parameter\u00b4s value of machine {}"
SHAPE_KEY = "shape parameter\u00b4s value of machine {}"


@dataclass(frozen=True)
class DatasetSpec:
    jobs: int = 50
    work_centers: int = 5
    machines_per_work_center: int = 3
    ddt: float = 0.5
    arrival_rate: int = 50
    initial_job_fraction: float = 0.5
    min_operations: int = 6
    max_operations: int = 10
    min_processing_time: int = 60
    max_processing_time: int = 120
    min_compatible_machines: int = 1
    max_compatible_machines: int | None = None

    @property
    def machines(self) -> int:
        return self.work_centers * self.machines_per_work_center


def _validate_spec(spec: DatasetSpec) -> None:
    if spec.jobs <= 0:
        raise ValueError("jobs must be positive")
    if spec.work_centers <= 0 or spec.machines_per_work_center <= 0:
        raise ValueError("work center and machine counts must be positive")
    if not 0 < spec.initial_job_fraction <= 1:
        raise ValueError("initial_job_fraction must be in (0, 1]")
    if spec.min_operations <= 0 or spec.max_operations < spec.min_operations:
        raise ValueError("operation bounds are invalid")
    if spec.min_processing_time <= 0 or spec.max_processing_time < spec.min_processing_time:
        raise ValueError("processing-time bounds are invalid")
    if spec.min_compatible_machines <= 0:
        raise ValueError("min_compatible_machines must be positive")
    max_compatible = spec.max_compatible_machines or spec.machines
    if max_compatible < spec.min_compatible_machines or max_compatible > spec.machines:
        raise ValueError("compatible-machine bounds are invalid")


def generate_dataset(path: str | Path, *, spec: DatasetSpec = DatasetSpec(), seed: int = 101) -> Path:
    """Generate a stable-ID .ini dataset accepted by the extracted environment."""

    _validate_spec(spec)
    rng = random.Random(seed)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    machine_ids = list(range(1, spec.machines + 1))
    max_compatible = spec.max_compatible_machines or spec.machines

    config = ConfigParser()
    config.add_section("world")
    world = config["world"]
    world["ddt"] = str(spec.ddt)
    world["number of jobs"] = str(spec.jobs)
    world["job arrival rate"] = str(spec.arrival_rate)

    for machine_id in machine_ids:
        scale = rng.randint(20, 40) * 60
        shape = round(rng.uniform(2.1, 2.7), 1)
        failure_cost = rng.randint(8000, 20000)
        cm_duration = rng.randint(480, 620)
        world[SCALE_KEY.format(machine_id)] = str(scale)
        world[SHAPE_KEY.format(machine_id)] = str(shape)
        world[f"failure cost of machine {machine_id}"] = str(failure_cost)
        world[f"corrective maintenance duration of machine {machine_id}"] = str(cm_duration)
        world[f"pm cost of machine {machine_id}"] = str(int(failure_cost / 3))
        world[f"pm duration of machine {machine_id}"] = str(int(cm_duration / 3))
        world[f"energy consumption of machine {machine_id}"] = str(round(rng.uniform(0.2, 0.9), 2))
        world[f"idle energy consumption of machine {machine_id}"] = str(round(rng.uniform(0.01, 0.02), 2))

    jobs = []
    machine_coverage = {machine_id: 0 for machine_id in machine_ids}
    for job_id in range(1, spec.jobs + 1):
        operation_count = rng.randint(spec.min_operations, spec.max_operations)
        operations: list[dict[int, int]] = []
        for _ in range(operation_count):
            compatible_count = rng.randint(spec.min_compatible_machines, max_compatible)
            compatible_machines = rng.sample(machine_ids, compatible_count)
            processing_times = {
                machine_id: rng.randint(spec.min_processing_time, spec.max_processing_time)
                for machine_id in compatible_machines
            }
            for machine_id in compatible_machines:
                machine_coverage[machine_id] += 1
            operations.append(processing_times)
        jobs.append(operations)

    for machine_id, count in machine_coverage.items():
        if count == 0:
            target_job = jobs[(machine_id - 1) % len(jobs)]
            target_operation = target_job[0]
            target_operation[machine_id] = rng.randint(spec.min_processing_time, spec.max_processing_time)

    initial_job_count = max(1, round(spec.jobs * spec.initial_job_fraction))
    initial_jobs = set(rng.sample(range(1, spec.jobs + 1), initial_job_count))
    arrival_time = 0.0

    for job_id, operations in enumerate(jobs, 1):
        world[f"operations of job {job_id}"] = str(len(operations))
        world[f"unit tardiness cost of job {job_id}"] = str(round(rng.uniform(0.3, 1.3), 1))
        if job_id in initial_jobs:
            job_arrival_time = 0
        else:
            arrival_time += rng.expovariate(1.0 / spec.arrival_rate)
            job_arrival_time = int(arrival_time)
        world[f"arrival time of job {job_id}"] = str(job_arrival_time)
        for operation_index, processing_times in enumerate(operations, 1):
            world[f"operation {operation_index} of job {job_id} available machines"] = str(dict(sorted(processing_times.items())))

    with output_path.open("w", encoding="utf-8") as handle:
        config.write(handle)
    return output_path
