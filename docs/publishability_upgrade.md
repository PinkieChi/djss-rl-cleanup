# Publishability Upgrade

This project now has the first pieces of a publishable experiment workflow: generated instance matrices with stable machine IDs, all dispatching-rule baselines, CSV results, Markdown summaries, confidence intervals, paired Wilcoxon comparisons against `SPT_DR_O`, and a held-out DQN generalization study.

## What Changed

- Added stable-ID dataset generation through `djss_rl.datasets`.
- Extended the dataset loader to accept generated machine IDs such as `{1: 42, 3: 57}` while preserving compatibility with the original memory-address dataset.
- Added `python -m djss_rl.cli generate-dataset`.
- Added `python -m djss_rl.cli experiment`.
- Added `python -m djss_rl.cli rl-study` for generated train/test instance studies.
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

## Next Publishable Step

Use these results as the baseline for improving the RL method itself. The next full paper-quality study should train with validation-based checkpointing, tune reward/state design, run more training seeds, and evaluate on larger held-out matrices:

```bash
python3 -m djss_rl.cli experiment --output-dir outputs/publishability-baseline-large --jobs-values 20,50,100 --ddt-values 0.5,1.0,1.5 --arrival-rates 50,100,200 --seeds 101,202,303,404,505
```

The `rl-study` command provides the held-out evaluation path. For a publishable claim, expand it to at least 10 training seeds per configuration and add a validation split so model selection is separated from final test reporting.
