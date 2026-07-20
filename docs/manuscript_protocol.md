# Manuscript Protocol

This document defines the next publishability-grade protocol for the project. It separates the current paper-ready claim from the stronger experiments needed for a journal-level submission.

## Current Defensible Claim

The strongest current claim is:

> A validation-selected dense-reward DQN significantly improves over the primary `SPT_DR_O` dispatching baseline on held-out dynamic job-shop instances.

The current evidence should not claim broad dominance over every dispatching rule or operating regime. ATC-style policies are best framed as a stress-test family, and the expanded 20/50/100-job checkpoint study shows that SPT remains the strongest broad baseline unless the DQN is retrained on the larger matrix.

## Add-On Generalization Study

The fastest add-on result is to evaluate the already-trained dense checkpoints on a broader unseen matrix. This does not retrain the agent, so it is a generalization check rather than a replacement for the main paper-study training protocol.

```bash
python3 -m djss_rl.cli checkpoint-study \
  --output-dir outputs/dense-broad-generalization-20260720 \
  --checkpoint-glob 'outputs/paper-study-20260719/dense/agents/seed-*/Best_agent*.pth' \
  --jobs-values 20,50,100 \
  --ddt-values 0.5,1.0,1.5 \
  --arrival-rates 50,100,200 \
  --test-instance-seeds 606,707,808,909,1001
```

This evaluates 10 dense checkpoints over 135 additional generated instances and compares them with all dispatching-rule baselines. The output files are:

- `outputs/dense-broad-generalization-20260720/checkpoint_results.csv`
- `outputs/dense-broad-generalization-20260720/checkpoint_summary.md`
- `outputs/dense-broad-generalization-20260720/checkpoint_study_config.json`

Completed July 20, 2026: all 2,565 rows completed with 0 failures. Overall mean tardiness ranked `SPT_DR_O` first (`0.320356`), `MRT_DR_O` second (`0.368362`), DQN third (`0.371964` across 1,350 checkpoint-instance rows), and `ATC_DR_O` fourth (`0.375809`). DQN beat ATC overall by a small but significant margin, but it lost to SPT on the broad matrix. Use this as evidence of partial external generalization and as motivation for the full retraining study.

## Full Retraining Study

For a stronger submission, retrain under the expanded matrix rather than only evaluating existing checkpoints:

```bash
python3 -m djss_rl.cli paper-study \
  --output-dir outputs/paper-study-expanded \
  --variants dense,dense_slow_epsilon,dense_low_lr,sharp \
  --jobs-values 20,50,100 \
  --ddt-values 0.5,1.0,1.5 \
  --arrival-rates 50,100,200 \
  --train-instance-seeds 101,202,303 \
  --validation-instance-seeds 505,606 \
  --test-instance-seeds 707,808,909,1001,1112 \
  --training-seeds 11,22,33,44,55,66,77,88,99,110 \
  --episodes 1000 \
  --validation-every 50
```

This is expensive, but it is the cleanest path to a stronger claim because training, validation, and final testing all scale together.

## Benchmark-Derived Instances

Reviewers are more comfortable when generated experiments are paired with known benchmark families. The new converter supports JSPLIB-style files:

```bash
python3 -m djss_rl.cli convert-jsplib \
  --input benchmarks/ft06.txt \
  --output outputs/benchmarks/ft06_dynamic.ini \
  --ddt 1.0 \
  --arrival-rate 50 \
  --initial-job-fraction 1.0
```

Use `--initial-job-fraction 1.0` for a static benchmark-style release of all jobs at time zero, or a lower value such as `0.5` to create a dynamic-arrival variant.

Useful benchmark families to include are `ft`, `la`, `abz`, `orb`, `swv`, `yn`, and `ta`. JobShopLib documents these as loadable benchmark families, and JSPLIB-style repositories provide plain-text versions of the same classic instances.

## Ablation Table

The manuscript should include an ablation table with at least:

| Ablation | Purpose |
|---|---|
| `sharp` reward | Original sparse/sharp reward baseline |
| `dense` reward | Dense tardiness-delta reward contribution |
| `dense_slow_epsilon` | Exploration-schedule sensitivity |
| `dense_low_lr` | Learning-rate sensitivity |
| no validation checkpointing | Shows why validation selection matters |
| 500, 1000, 2000 episodes | Training-budget sensitivity |

The current `paper-study` command covers the first four. A future no-validation variant and episode-budget sweep would make the ablation section stronger.

## Reporting Checklist

Report all final results using held-out test rows only:

- mean, median, standard deviation, and 95% confidence interval
- paired Wilcoxon test versus the primary SPT baseline
- win/loss/tie counts
- per-regime breakdown by job count, due-date tightness, and arrival rate
- runtime or makespan comparison
- validation-selection method and seed list
- raw CSV and config paths

The abstract should headline the statistically significant SPT result. The limitations section should say that ATC-style due-date-aware dispatching remains an important future-work target.
