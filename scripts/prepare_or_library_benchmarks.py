"""Prepare OR-Library job-shop benchmarks for this DJSS environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from djss_rl.datasets import generate_dataset_from_jsplib


DEFAULT_INSTANCES = [
    "ft06",
    "ft10",
    "ft20",
    "la01",
    "la06",
    "la21",
    "la31",
    "orb01",
    "orb02",
    "abz5",
    "abz7",
    "swv01",
    "swv06",
    "yn1",
    "yn2",
]

SOURCE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt"


def _is_int_pair(line: str) -> bool:
    parts = line.split()
    if len(parts) != 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True


def extract_instances(source_path: Path) -> dict[str, list[str]]:
    """Return JSPLIB-style text blocks from OR-Library jobshop1.txt."""

    lines = source_path.read_text(encoding="utf-8").splitlines()
    instances: dict[str, list[str]] = {}
    index = 0
    while index < len(lines):
        clean_line = lines[index].strip()
        if not clean_line.startswith("instance "):
            index += 1
            continue

        name = clean_line.split()[1]
        dims_index = index + 1
        while dims_index < len(lines) and not _is_int_pair(lines[dims_index].strip()):
            dims_index += 1
        if dims_index >= len(lines):
            raise ValueError(f"Could not find dimensions for instance {name}")

        jobs, machines = (int(value) for value in lines[dims_index].split())
        block = [f"# OR-Library jobshop1 instance {name}", f"{jobs} {machines}"]
        data_start = dims_index + 1
        data_end = data_start + jobs
        if data_end > len(lines):
            raise ValueError(f"Instance {name} is truncated")
        for row in lines[data_start:data_end]:
            values = row.split()
            if len(values) < machines * 2:
                raise ValueError(f"Instance {name} has a short operation row: {row}")
            block.append(" ".join(values[: machines * 2]))
        instances[name] = block
        index = data_end
    return instances


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="benchmarks/or-library/raw/jobshop1.txt")
    parser.add_argument("--instance-dir", default="benchmarks/or-library/instances")
    parser.add_argument("--output-dir", default="outputs/or-library-benchmark-derived/datasets")
    parser.add_argument("--instances", default=",".join(DEFAULT_INSTANCES))
    parser.add_argument("--ddt-values", default="0.5,1.0")
    parser.add_argument("--arrival-rates", default="50,100")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--initial-job-fraction", type=float, default=0.5)
    args = parser.parse_args()

    source_path = Path(args.source)
    instance_dir = Path(args.instance_dir)
    output_dir = Path(args.output_dir)
    selected_names = _parse_str_list(args.instances)
    ddt_values = _parse_float_list(args.ddt_values)
    arrival_rates = _parse_int_list(args.arrival_rates)

    extracted = extract_instances(source_path)
    missing = [name for name in selected_names if name not in extracted]
    if missing:
        raise ValueError(f"Unknown OR-Library instances: {', '.join(missing)}")

    instance_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for name in selected_names:
        instance_path = instance_dir / f"{name}.txt"
        instance_path.write_text("\n".join(extracted[name]) + "\n", encoding="utf-8")
        for ddt in ddt_values:
            for arrival_rate in arrival_rates:
                converted_path = output_dir / f"{name}_ddt{ddt:g}_arr{arrival_rate}_seed{args.seed}.ini"
                generate_dataset_from_jsplib(
                    instance_path,
                    converted_path,
                    ddt=ddt,
                    arrival_rate=arrival_rate,
                    initial_job_fraction=args.initial_job_fraction,
                    seed=args.seed,
                    machine_index_base=0,
                )
                manifest_rows.append(
                    {
                        "instance": name,
                        "source_path": str(instance_path),
                        "converted_path": str(converted_path),
                        "ddt": ddt,
                        "arrival_rate": arrival_rate,
                        "seed": args.seed,
                        "initial_job_fraction": args.initial_job_fraction,
                    }
                )

    manifest = {
        "source_url": SOURCE_URL,
        "source_file": str(source_path),
        "instances": selected_names,
        "converted_datasets": manifest_rows,
    }
    (output_dir.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("prepared_instances", len(selected_names))
    print("converted_datasets", len(manifest_rows))
    print("manifest", output_dir.parent / "manifest.json")


if __name__ == "__main__":
    main()
