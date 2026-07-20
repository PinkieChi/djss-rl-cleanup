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


def parse_jsplib(path: str | Path, *, machine_index_base: int = 0) -> list[list[tuple[int, int]]]:
    """Parse a JSPLIB-style job-shop benchmark file.

    The expected format is:

    ``jobs machines``
    followed by ``machine duration`` pairs for every operation of every job.
    Machine IDs are converted to the 1-based IDs used by this project.
    """

    input_path = Path(path)
    numbers: list[int] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            clean_line = line.split("#", 1)[0].strip()
            if not clean_line:
                continue
            try:
                numbers.extend(int(item) for item in clean_line.split())
            except ValueError:
                # Many benchmark files include prose headers before the numeric matrix.
                continue

    if len(numbers) < 2:
        raise ValueError("JSPLIB file must start with '<jobs> <machines>'")

    jobs_count, machines_count = numbers[0], numbers[1]
    if jobs_count <= 0 or machines_count <= 0:
        raise ValueError("JSPLIB jobs and machines must be positive")

    expected_values = jobs_count * machines_count * 2
    values = numbers[2:]
    if len(values) < expected_values:
        raise ValueError(
            f"JSPLIB file has {len(values)} operation values, expected at least {expected_values}"
        )

    jobs: list[list[tuple[int, int]]] = []
    cursor = 0
    for _ in range(jobs_count):
        operations: list[tuple[int, int]] = []
        for _ in range(machines_count):
            raw_machine_id = values[cursor]
            duration = values[cursor + 1]
            machine_id = raw_machine_id - machine_index_base + 1
            if not 1 <= machine_id <= machines_count:
                raise ValueError(f"machine ID {raw_machine_id} is outside the benchmark machine range")
            if duration <= 0:
                raise ValueError("processing durations must be positive")
            operations.append((machine_id, duration))
            cursor += 2
        jobs.append(operations)
    return jobs


def generate_dataset_from_jsplib(
    input_path: str | Path,
    output_path: str | Path,
    *,
    ddt: float = 1.0,
    arrival_rate: int = 50,
    initial_job_fraction: float = 1.0,
    seed: int = 101,
    machine_index_base: int = 0,
) -> Path:
    """Convert a JSPLIB-style benchmark into this project's dynamic `.ini` format."""

    jobs = parse_jsplib(input_path, machine_index_base=machine_index_base)
    if not 0 < initial_job_fraction <= 1:
        raise ValueError("initial_job_fraction must be in (0, 1]")

    rng = random.Random(seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    jobs_count = len(jobs)
    machines_count = max(machine_id for job in jobs for machine_id, _ in job)

    config = ConfigParser()
    config.add_section("world")
    world = config["world"]
    world["ddt"] = str(ddt)
    world["number of jobs"] = str(jobs_count)
    world["job arrival rate"] = str(arrival_rate)

    for machine_id in range(1, machines_count + 1):
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

    initial_job_count = max(1, round(jobs_count * initial_job_fraction))
    initial_jobs = set(rng.sample(range(1, jobs_count + 1), initial_job_count))
    arrival_time = 0.0
    for job_id, operations in enumerate(jobs, 1):
        world[f"operations of job {job_id}"] = str(len(operations))
        world[f"unit tardiness cost of job {job_id}"] = str(round(rng.uniform(0.3, 1.3), 1))
        if job_id in initial_jobs:
            job_arrival_time = 0
        else:
            arrival_time += rng.expovariate(1.0 / arrival_rate)
            job_arrival_time = int(arrival_time)
        world[f"arrival time of job {job_id}"] = str(job_arrival_time)
        for operation_index, (machine_id, duration) in enumerate(operations, 1):
            world[f"operation {operation_index} of job {job_id} available machines"] = str({machine_id: duration})

    with output.open("w", encoding="utf-8") as handle:
        config.write(handle)
    return output
