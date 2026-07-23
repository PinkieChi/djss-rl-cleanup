# Reproducibility Report

## Artifact Summary

This repository contains a cleaned implementation of a DQN-based dynamic job shop scheduling project. The original notebook is preserved as a research narrative, while the core environment, agent, training, evaluation, dataset generation, benchmark conversion, and experiment tooling have been extracted into the `djss_rl` package.

The package answers a narrow reproducibility question:

> Can the original DQN scheduling project be executed, evaluated, extended to held-out generated and benchmark-derived instances, and interpreted against strong dispatching-rule baselines?

The answer is yes. The stronger scientific answer is cautious: the DQN pipeline is reproducible and competitive, but it does not establish broad superiority over dispatching rules.

## Main Reproducibility Improvements

- Safe notebook defaults: W&B, training, and expensive evaluation are opt-in.
- Restored valid `.ini` dataset artifact.
- Importable Python modules under `djss_rl/`.
- CLI commands for smoke tests, evaluation, training, generated datasets, JSPLIB conversion, OR-Library preparation, baseline matrices, RL studies, paper studies, benchmark studies, checkpoint generalization, and policy tracing.
- Unit tests covering notebook loading, environment construction, baseline scheduling, checkpoint evaluation, generated datasets, JSPLIB conversion, study writers, benchmark study writers, policy tracing, and simulator deadlock reporting.
- Markdown and CSV outputs for experiment results.

## Core Commands

Smoke test:

```bash
python3 -m djss_rl.cli smoke
```

Expanded generated-instance DQN study:

```bash
python3 -m djss_rl.cli paper-study \
  --output-dir outputs/expanded-dense-multiseed-20260721 \
  --variants dense \
  --jobs-values 20,50,100 \
  --ddt-values 0.5,1.0,1.5 \
  --arrival-rates 50,100,200 \
  --train-instance-seeds 101,202 \
  --validation-instance-seeds 505,606 \
  --test-instance-seeds 707,808,909 \
  --training-seeds 11,22,33 \
  --episodes 500 \
  --validation-every 50
```

OR-Library benchmark-derived study:

```bash
python3 scripts/prepare_or_library_benchmarks.py \
  --source benchmarks/or-library/raw/jobshop1.txt \
  --output-dir outputs/or-library-benchmark-derived-20260721/datasets

python3 -m djss_rl.cli benchmark-study \
  --output-dir outputs/or-library-benchmark-study-20260723-clean \
  --dataset-glob 'outputs/or-library-benchmark-derived-20260721/datasets/*.ini' \
  --checkpoint-glob 'outputs/expanded-dense-multiseed-20260721/dense/agents/seed-*/Best_agent*.pth'
```

Multi-checkpoint policy trace:

```bash
python3 -m djss_rl.cli trace-checkpoints \
  --output-dir outputs/or-library-benchmark-study-20260723-clean/policy-trace \
  --checkpoint-glob 'outputs/expanded-dense-multiseed-20260721/dense/agents/seed-*/Best_agent*.pth' \
  --dataset-glob 'outputs/or-library-benchmark-derived-20260721/datasets/*.ini'
```

## Key Results

### Expanded Generated-Instance Retraining

The expanded dense DQN study trained 3 independent checkpoints for 500 episodes each, using 54 training instances, 54 validation instances, and 81 held-out generated test instances.

| Method | Mean tardiness | Rank |
|---|---:|---:|
| `SPT_DR_O` | 0.323933 | 1 |
| DQN | 0.340194 | 2 |
| `MRT_DR_O` | 0.372714 | 3 |
| `ATC_DR_O` | 0.378592 | 4 |

Against `SPT_DR_O`, DQN was worse on average by `+0.016260` tardiness. Against `ATC_DR_O`, DQN was better on average by `-0.038398`.

### Benchmark-Derived Generalization

The OR-Library-derived benchmark study converted 15 named job-shop instances into 60 dynamic `.ini` datasets. It evaluated all 9 dispatching rules and 3 DQN checkpoints.

| Method | Mean tardiness | Rank |
|---|---:|---:|
| `SPT_DR_O` | 0.514833 | 1 |
| DQN | 0.524708 | 2 |
| `ATC_DR_O` | 0.528806 | 3 |
| `MRT_DR_O` | 0.544458 | 4 |

The result table contains 720 rows: 718 successful rows and 2 explicit `LRT_DR_O` simulator deadlock rows on `abz7` arrival-rate-50 variants.

### Policy Diagnostics

Policy traces show that the selected DQN checkpoints behave like deterministic single-rule policies:

| Training seed | Generated dominant action | Benchmark dominant action |
|---:|---|---|
| 11 | `SPT_DR_O` at 100% | `SPT_DR_O` at 100% |
| 22 | `SPT_DR_O` at 100% | `SPT_DR_O` at 100% |
| 33 | `MRT_DR_O` at 100% | `MRT_DR_O` at 100% |

This is the key reproducibility finding: the DQN pipeline runs and is competitive, but the learned policy does not yet show adaptive rule mixing.

## Recommended Interpretation

The strongest interpretation is:

> Dense rewards and validation checkpointing make DQN dispatching-rule selection reproducible and competitive, but broader generated and benchmark-derived evaluation shows that `SPT_DR_O` remains stronger overall. Policy tracing is essential because the selected DQN checkpoints collapse to single classical dispatching rules.

## Reviewer-Safe Limitations

- The restored source dataset contains reconstructed machine identity because the original file encoded compatible machines as memory-address tokens.
- The DQN action space is a menu of dispatching rules, so learned behavior can become rule selection rather than new scheduling logic.
- Expanded experiments used 3 DQN training seeds and 500 episodes; a larger compute budget may be needed for final algorithmic claims.
- Two converted benchmark rows exposed simulator deadlocks for `LRT_DR_O`; these are reported explicitly.
- Strong baselines such as `SPT_DR_O` and `ATC_DR_O` remain competitive or superior in important regimes.
- No real-factory, MES, or digital-twin validation is included.

## Recommended Next Experiment

Use this project as a transparent DQN baseline and diagnostic artifact. For a stronger RA-PPO paper, test whether the new method avoids the single-rule collapse shown here and learns adaptive action mixtures across job counts, due-date tightness, arrival rates, and benchmark-derived regimes.
