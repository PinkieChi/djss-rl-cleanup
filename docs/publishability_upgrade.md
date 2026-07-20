# Publishability Upgrade

This project now has the first pieces of a publishable experiment workflow: generated instance matrices with stable machine IDs, all dispatching-rule baselines, CSV results, Markdown summaries, confidence intervals, paired Wilcoxon comparisons, validation-based checkpoint selection, and held-out DQN generalization studies.

## What Changed

- Added stable-ID dataset generation through `djss_rl.datasets`.
- Extended the dataset loader to accept generated machine IDs such as `{1: 42, 3: 57}` while preserving compatibility with the original memory-address dataset.
- Added `python -m djss_rl.cli generate-dataset`.
- Added `python -m djss_rl.cli experiment`.
- Added `python -m djss_rl.cli rl-study` for generated train/test instance studies.
- Added `python -m djss_rl.cli paper-study` for resumable multi-variant studies.
- Added `python -m djss_rl.cli checkpoint-study` for broad held-out evaluation of existing checkpoints.
- Added `python -m djss_rl.cli convert-jsplib` for JSPLIB-style benchmark conversion.
- Added optional validation splits, dense tardiness reward shaping, and tunable DQN hyperparameters.
- Added experiment summaries with mean tardiness, standard deviation, 95% confidence intervals, and paired comparison tables.
- Added tests for generated dataset loading, tiny experiment-grid execution, and tiny held-out RL-study execution.

## Pilot Matrix

The pilot run used 8 generated dynamic instances:

```bash
python3 -m djss_rl.cli experiment --output-dir outputs/publishability-baseline-20260718 --jobs-values 12 --ddt-values 0.5,1.0 --arrival-rates 50,100 --seeds 11,22 --work-centers 2 --machines-per-work-center 2 --min-operations 3 --max-operations 5 --min-processing-time 20 --max-processing-time 60
```

The local outputs are:

- `outputs/publishability-baseline-20260718/results.csv`
- `outputs/publishability-baseline-20260718/summary.md`
- `outputs/publishability-baseline-20260718/datasets/*.ini`

## Pilot Result

| Rank | Method | n | Mean tardiness | Std | 95% CI | Mean makespan | Mean utilization |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | SPT_DR_O | 8 | 0.368116 | 0.188063 | 0.130321 | 652.750000 | 69.418688% |
| 2 | MRT_DR_O | 8 | 0.418056 | 0.178407 | 0.123630 | 661.000000 | 69.066221% |
| 3 | ATC_DR_O | 8 | 0.420531 | 0.185660 | 0.128656 | 655.750000 | 68.370711% |
| 4 | LRT_DR_O | 8 | 0.425845 | 0.192236 | 0.133213 | 681.250000 | 67.406498% |
| 5 | EDD_DR_O | 8 | 0.450725 | 0.198576 | 0.137606 | 670.875000 | 67.855628% |
| 6 | SLK_DR_O | 8 | 0.464614 | 0.202183 | 0.140106 | 663.000000 | 70.244861% |
| 7 | LSPO_DR_O | 8 | 0.470229 | 0.194816 | 0.135001 | 656.875000 | 71.467425% |
| 8 | MCR_DR_O | 8 | 0.475302 | 0.213981 | 0.148281 | 678.250000 | 68.654362% |
| 9 | CR_DR_O | 8 | 0.489070 | 0.194042 | 0.134464 | 682.250000 | 68.051085% |

Paired against `SPT_DR_O`, the closest competitor was `MRT_DR_O` with mean tardiness difference `+0.049940` and Wilcoxon `p = 0.062500`. All other rules were worse than SPT by larger average margins in this pilot.

## Interpretation

This pilot is not publication evidence yet because it is intentionally small. It is a validation of the experiment machinery. The pattern is useful: `SPT_DR_O` remains a strong baseline, so any RL method should be judged against SPT across a broad generated matrix, not only against weaker rules.

## Completed Larger Runs

The larger baseline matrix and held-out DQN study have now been run. A concise record is available in `docs/publishability_results.md`.

Key result:

- The 135-instance baseline matrix found `SPT_DR_O` had the best average tardiness: `0.320584`.
- The held-out DQN study trained 3 seeds for 1,000 episodes each and evaluated them on 8 held-out instances.
- DQN mean held-out tardiness was `0.282789`, close to `SPT_DR_O` at `0.273815`, but not better. The paired comparison against SPT had Wilcoxon `p = 0.224728`.
- A validation-selected dense-reward DQN study improved DQN mean held-out tardiness to `0.277797`.
- A 4-variant paper study then trained 40 DQN checkpoints with 10 training seeds per variant. The best dense-reward variant reached mean held-out tardiness `0.267804`, significantly better than the primary `SPT_DR_O` baseline at `0.273815` (`p = 0.002808`).
- A broad checkpoint generalization study then evaluated the dense checkpoints on 135 additional held-out generated instances. DQN ranked third overall with mean tardiness `0.371964`, slightly beating `ATC_DR_O` (`0.375809`) but not the strongest broad baseline, `SPT_DR_O` (`0.320356`).

## Next Publishable Step

Use these results as the baseline for improving and reporting the RL method itself. The current evidence supports a careful paper claim that a validation-selected dense-reward DQN significantly improves over SPT on the selected held-out RL matrix. The next full paper-quality extension should retrain on the larger generated matrix rather than only evaluating existing checkpoints:

```bash
python3 -m djss_rl.cli paper-study --output-dir outputs/paper-study-expanded --variants dense,dense_slow_epsilon,dense_low_lr,sharp --jobs-values 20,50,100 --ddt-values 0.5,1.0,1.5 --arrival-rates 50,100,200 --train-instance-seeds 101,202,303 --validation-instance-seeds 505,606 --test-instance-seeds 707,808,909,1001,1112 --training-seeds 11,22,33,44,55,66,77,88,99,110 --episodes 1000 --validation-every 50
```

The `paper-study` command now provides the resumable held-out evaluation path with at least 10 training seeds per configuration, validation-based model selection, and reward/hyperparameter variant comparison. The detailed next protocol is recorded in `docs/manuscript_protocol.md`.
