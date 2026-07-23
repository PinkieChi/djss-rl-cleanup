# Journal Readiness Results - July 23, 2026

This document records the stronger experiment set requested for a journal-facing assessment:

- expanded-matrix DQN retraining with multiple training seeds;
- benchmark-derived instances from OR-Library job-shop problems;
- transparent comparison against all dispatching-rule baselines, including `ATC_DR_O`;
- checkpoint policy-trace diagnostics;
- an interpretation that positions the DQN results as motivation for stronger RL designs such as RA-PPO.

## Bottom Line

The project is much stronger than the initial notebook, but it is not yet a full journal paper claiming that DQN is superior to classical dispatching rules.

The best defensible claim is:

> Dense reward shaping and validation checkpointing make the DQN pipeline reproducible and competitive, but the selected DQN policies mostly collapse to single classical dispatching rules. The evidence is strongest as a diagnostic study and as motivation for stronger adaptive RL designs such as RA-PPO.

## Expanded Generated-Instance Retraining

Command:

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

Protocol:

- 3 independent DQN training seeds.
- 54 generated training instances.
- 54 generated validation instances.
- 81 held-out generated test instances.
- Dense tardiness reward and validation-selected checkpointing.

Main result:

| Method | Mean tardiness | Rank |
|---|---:|---:|
| `SPT_DR_O` | 0.323933 | 1 |
| DQN | 0.340194 | 2 |
| `MRT_DR_O` | 0.372714 | 3 |
| `ATC_DR_O` | 0.378592 | 4 |

Against `SPT_DR_O`, DQN was worse on average by `+0.016260` tardiness, with Wilcoxon `p = 0.000000` and win/loss/tie counts of `9/61/173`.

Against `ATC_DR_O`, DQN was better on average by `-0.038398` tardiness, with Wilcoxon `p = 0.000000`.

Interpretation: DQN is competitive and beats most weaker rules, but `SPT_DR_O` remains the strongest generated-instance baseline on the expanded matrix.

## Generated-Instance Policy Trace

Command:

```bash
python3 -m djss_rl.cli trace-checkpoints \
  --output-dir outputs/expanded-dense-multiseed-20260721/dense/policy-trace \
  --checkpoint-glob 'outputs/expanded-dense-multiseed-20260721/dense/agents/seed-*/Best_agent*.pth' \
  --dataset-glob 'outputs/expanded-dense-multiseed-20260721/dense/test/datasets/*.ini'
```

Result:

| Training seed | Test instances | Decisions | Mean tardiness | Dominant action | Dominant fraction |
|---:|---:|---:|---:|---|---:|
| 11 | 81 | 37,227 | 0.323933 | `SPT_DR_O` | 1.000000 |
| 22 | 81 | 37,227 | 0.323933 | `SPT_DR_O` | 1.000000 |
| 33 | 81 | 37,374 | 0.372714 | `MRT_DR_O` | 1.000000 |

Interpretation: the best DQN checkpoints are not learning a rich adaptive mixture. Seeds 11 and 22 reproduce `SPT_DR_O`; seed 33 reproduces `MRT_DR_O`.

## Benchmark-Derived OR-Library Study

Benchmark source:

- OR-Library job-shop page: <https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html>
- Local source notes: `benchmarks/or-library/README.md`

Preparation command:

```bash
python3 scripts/prepare_or_library_benchmarks.py \
  --source benchmarks/or-library/raw/jobshop1.txt \
  --output-dir outputs/or-library-benchmark-derived-20260721/datasets
```

Evaluation command:

```bash
python3 -m djss_rl.cli benchmark-study \
  --output-dir outputs/or-library-benchmark-study-20260723-clean \
  --dataset-glob 'outputs/or-library-benchmark-derived-20260721/datasets/*.ini' \
  --checkpoint-glob 'outputs/expanded-dense-multiseed-20260721/dense/agents/seed-*/Best_agent*.pth'
```

Protocol:

- 15 named OR-Library instances: `ft06`, `ft10`, `ft20`, `la01`, `la06`, `la21`, `la31`, `orb01`, `orb02`, `abz5`, `abz7`, `swv01`, `swv06`, `yn1`, `yn2`.
- 4 dynamic variants per named instance from due-date tightness and arrival-rate settings.
- 60 converted benchmark-derived `.ini` datasets.
- All 9 dispatching-rule baselines plus 3 DQN checkpoints.
- 720 total rows: 718 successful rows and 2 explicit simulator deadlock rows for `LRT_DR_O` on `abz7` arrival-rate-50 variants.

Main result:

| Rank | Method | n | Mean tardiness |
|---:|---|---:|---:|
| 1 | `SPT_DR_O` | 60 | 0.514833 |
| 2 | DQN | 180 | 0.524708 |
| 3 | `ATC_DR_O` | 60 | 0.528806 |
| 4 | `MRT_DR_O` | 60 | 0.544458 |
| 5 | `LRT_DR_O` | 58 | 0.547802 |
| 6 | `EDD_DR_O` | 60 | 0.559417 |
| 7 | `LSPO_DR_O` | 60 | 0.571111 |
| 8 | `SLK_DR_O` | 60 | 0.571111 |
| 9 | `MCR_DR_O` | 60 | 0.571625 |
| 10 | `CR_DR_O` | 60 | 0.591032 |

DQN against key baselines:

| Baseline | Paired rows | Mean DQN-baseline difference | Wilcoxon p | Wins | Losses | Ties |
|---|---:|---:|---:|---:|---:|---:|
| `SPT_DR_O` | 180 | +0.009875 | 0.000005 | 13 | 40 | 127 |
| `ATC_DR_O` | 180 | -0.004097 | 0.001686 | 98 | 46 | 36 |
| `MRT_DR_O` | 180 | -0.019750 | 0.000000 | 80 | 26 | 74 |
| `CR_DR_O` | 180 | -0.066324 | 0.000000 | 168 | 5 | 7 |

Interpretation: benchmark-derived results confirm the generated-instance pattern. DQN is competitive and slightly better than `ATC_DR_O` on average, but `SPT_DR_O` remains the strongest method overall.

## Benchmark Policy Trace

Command:

```bash
python3 -m djss_rl.cli trace-checkpoints \
  --output-dir outputs/or-library-benchmark-study-20260723-clean/policy-trace \
  --checkpoint-glob 'outputs/expanded-dense-multiseed-20260721/dense/agents/seed-*/Best_agent*.pth' \
  --dataset-glob 'outputs/or-library-benchmark-derived-20260721/datasets/*.ini'
```

Result:

| Training seed | Benchmark datasets | Decisions | Mean tardiness | Dominant action | Dominant fraction |
|---:|---:|---:|---:|---|---:|
| 11 | 60 | 10,844 | 0.514833 | `SPT_DR_O` | 1.000000 |
| 22 | 60 | 10,844 | 0.514833 | `SPT_DR_O` | 1.000000 |
| 33 | 60 | 10,844 | 0.544458 | `MRT_DR_O` | 1.000000 |

Interpretation: the benchmark trace confirms that the DQN checkpoints behave as single-rule selectors across external benchmark-derived instances too.

## What Changed Technically

- Added `benchmark-study` CLI support for evaluating converted benchmark datasets.
- Added `trace-checkpoints` CLI support for multi-checkpoint policy diagnostics.
- Added OR-Library preparation tooling and benchmark documentation.
- Added benchmark-study and checkpoint-trace tests.
- Fixed observation handling for empty ready-machine buffers.
- Added a simulator deadlock guard so no-event states become explicit experiment failures instead of infinite runs.

## Publishability Assessment

This is not yet enough for a full journal paper claiming DQN superiority.

It can support a stronger, honest paper or thesis chapter framed as:

- a reproducible restoration of a DQN-based DJSS project;
- a transparent benchmark of DQN rule selection against strong dispatching rules;
- evidence that naive DQN often collapses to one classical rule;
- motivation for richer RL designs such as RA-PPO.

To make the next RA-PPO paper stronger, use this DQN project as the baseline/negative evidence and show that RA-PPO addresses the specific weakness exposed here: lack of adaptive action mixing across operating regimes.

## Recommended Manuscript Framing

Use this phrasing:

> The DQN agent is competitive with strong dispatching heuristics and outperforms several weaker rules, but expanded generated-instance and benchmark-derived experiments show that it does not consistently outperform `SPT_DR_O`. Policy-trace diagnostics reveal that selected checkpoints collapse to deterministic single-rule behavior, mainly `SPT_DR_O` or `MRT_DR_O`. These findings motivate adaptive policy-gradient approaches such as RA-PPO, which should be evaluated for regime-aware rule mixing and stronger generalization.

Avoid this phrasing:

> The proposed DQN method outperforms classical dispatching rules.

That claim is not supported by the current results.
