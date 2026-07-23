# Publishability Experiment Results

This document records the larger generated-instance experiments run on July 18-20, 2026. The July 23, 2026 expanded retraining and benchmark-derived results supersede the publication conclusion here; see `docs/journal_readiness_results_20260723.md` for the current journal-readiness assessment. The raw CSV, logs, generated datasets, and trained checkpoints are stored under `outputs/` locally and are packaged separately for backup.

## Large Baseline Matrix

Command:

```bash
python3 -m djss_rl.cli experiment --output-dir outputs/publishability-baseline-large-20260718 --jobs-values 20,50,100 --ddt-values 0.5,1.0,1.5 --arrival-rates 50,100,200 --seeds 101,202,303,404,505
```

This evaluated 135 generated dynamic job-shop instances with 9 dispatching-rule baselines. All 1,215 method-instance rows completed successfully.

| Rank | Method | n | Mean tardiness | Std | 95% CI | Mean makespan | Mean utilization |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | SPT_DR_O | 135 | 0.320584 | 0.238081 | 0.040162 | 4585.377778 | 57.350661% |
| 2 | MRT_DR_O | 135 | 0.369819 | 0.237454 | 0.040056 | 4631.377778 | 62.575786% |
| 3 | ATC_DR_O | 135 | 0.378242 | 0.281520 | 0.047490 | 4557.733333 | 58.637027% |
| 4 | LRT_DR_O | 135 | 0.388349 | 0.271563 | 0.045810 | 4790.111111 | 58.855497% |
| 5 | SLK_DR_O | 135 | 0.431089 | 0.308813 | 0.052094 | 4708.481481 | 60.858216% |
| 6 | EDD_DR_O | 135 | 0.434547 | 0.311357 | 0.052523 | 4749.481481 | 59.698233% |
| 7 | LSPO_DR_O | 135 | 0.435092 | 0.312586 | 0.052730 | 4721.370370 | 60.581888% |
| 8 | MCR_DR_O | 135 | 0.438199 | 0.316065 | 0.053317 | 4749.800000 | 59.972439% |
| 9 | CR_DR_O | 135 | 0.454033 | 0.311745 | 0.052588 | 4692.266667 | 60.063245% |

Paired Wilcoxon comparisons against `SPT_DR_O` found every other baseline had higher mean tardiness with `p < 0.001`. The closest rule was `MRT_DR_O`, with mean tardiness `+0.049235` higher than SPT.

## Held-Out DQN Study

Command:

```bash
python3 -m djss_rl.cli rl-study --output-dir outputs/rl-generalization-20260718 --jobs-values 20 --ddt-values 0.5,1.0 --arrival-rates 50,100 --train-instance-seeds 101,202 --test-instance-seeds 303,404 --training-seeds 11,22,33 --episodes 1000
```

This trained 3 independent DQN checkpoints for 1,000 episodes each on 8 generated training instances, then evaluated them on 8 held-out generated test instances. All 96 test rows completed successfully.

| Training seed | Episodes | Best training score |
|---:|---:|---:|
| 11 | 1000 | 43.5 |
| 22 | 1000 | 40.75 |
| 33 | 1000 | 43.5 |

Test ranking:

| Rank | Method | n | Mean tardiness | Std | 95% CI | Mean makespan | Mean utilization |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ATC_DR_O | 8 | 0.266695 | 0.233715 | 0.161956 | 1504.875000 | 62.739723% |
| 2 | SPT_DR_O | 8 | 0.273815 | 0.224499 | 0.155570 | 1510.250000 | 62.532371% |
| 3 | SLK_DR_O | 8 | 0.282543 | 0.241066 | 0.167050 | 1521.125000 | 63.153824% |
| 4 | DQN | 24 | 0.282789 | 0.220537 | 0.088233 | 1521.666667 | 63.614522% |
| 5 | LRT_DR_O | 8 | 0.283339 | 0.242643 | 0.168143 | 1591.000000 | 60.317549% |
| 6 | LSPO_DR_O | 8 | 0.290489 | 0.245108 | 0.169851 | 1539.875000 | 62.592743% |
| 7 | MCR_DR_O | 8 | 0.291275 | 0.243959 | 0.169055 | 1531.000000 | 62.283247% |
| 8 | MRT_DR_O | 8 | 0.292883 | 0.222949 | 0.154496 | 1517.250000 | 64.168158% |
| 9 | EDD_DR_O | 8 | 0.294460 | 0.237167 | 0.164348 | 1559.625000 | 61.370243% |
| 10 | CR_DR_O | 8 | 0.314264 | 0.248298 | 0.172062 | 1569.250000 | 61.730648% |

DQN by training seed:

| Training seed | Held-out instances | Mean tardiness | Std |
|---:|---:|---:|---:|
| 11 | 8 | 0.269013 | 0.218773 |
| 22 | 8 | 0.292883 | 0.222949 |
| 33 | 8 | 0.286473 | 0.248776 |

Against `SPT_DR_O`, DQN had a mean tardiness difference of `+0.008975` across 24 paired comparisons, Wilcoxon `p = 0.224728`, with 9 wins, 11 losses, and 4 ties.

## Validation-Selected Dense-Reward DQN Study

Command:

```bash
python3 -m djss_rl.cli rl-study --output-dir outputs/rl-validation-dense-20260718 --jobs-values 20 --ddt-values 0.5,1.0 --arrival-rates 50,100 --train-instance-seeds 101,202 --validation-instance-seeds 505 --test-instance-seeds 303,404 --training-seeds 44,55,66 --episodes 500 --validation-every 50 --reward-mode dense_tardiness --train-start 500
```

This study added a separate validation split and selected each checkpoint by validation tardiness rather than training reward. It used a dense tardiness-delta reward and trained 3 independent seeds for 500 episodes each. All 144 result rows completed successfully: 48 validation rows and 96 held-out test rows.

| Training seed | Episodes | Best training score | Best validation tardiness |
|---:|---:|---:|---:|
| 44 | 500 | 6.691530 | 0.237952 |
| 55 | 500 | 6.310180 | 0.254518 |
| 66 | 500 | 6.681602 | 0.251506 |

Held-out test ranking:

| Rank | Method | n | Mean tardiness | Std | 95% CI | Mean makespan | Mean utilization |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ATC_DR_O | 8 | 0.266695 | 0.233715 | 0.161956 | 1504.875000 | 62.739723% |
| 2 | SPT_DR_O | 8 | 0.273815 | 0.224499 | 0.155570 | 1510.250000 | 62.532371% |
| 3 | DQN | 24 | 0.277797 | 0.217299 | 0.086938 | 1510.791667 | 63.146751% |
| 4 | SLK_DR_O | 8 | 0.282543 | 0.241066 | 0.167050 | 1521.125000 | 63.153824% |
| 5 | LRT_DR_O | 8 | 0.283339 | 0.242643 | 0.168143 | 1591.000000 | 60.317549% |
| 6 | LSPO_DR_O | 8 | 0.290489 | 0.245108 | 0.169851 | 1539.875000 | 62.592743% |
| 7 | MCR_DR_O | 8 | 0.291275 | 0.243959 | 0.169055 | 1531.000000 | 62.283247% |
| 8 | MRT_DR_O | 8 | 0.292883 | 0.222949 | 0.154496 | 1517.250000 | 64.168158% |
| 9 | EDD_DR_O | 8 | 0.294460 | 0.237167 | 0.164348 | 1559.625000 | 61.370243% |
| 10 | CR_DR_O | 8 | 0.314264 | 0.248298 | 0.172062 | 1569.250000 | 61.730648% |

DQN improved from the previous held-out mean tardiness `0.282789` to `0.277797`. It beat weaker baselines such as `CR_DR_O`, `EDD_DR_O`, `MRT_DR_O`, `MCR_DR_O`, and `LSPO_DR_O` in paired tests, but it still did not beat `SPT_DR_O` or `ATC_DR_O`. Against `SPT_DR_O`, DQN had mean difference `+0.003983`, Wilcoxon `p = 0.427246`, with 5 wins, 7 losses, and 12 ties.

## Multi-Variant Paper Study

Command:

```bash
python3 -m djss_rl.cli paper-study --output-dir outputs/paper-study-20260719 --variants dense,sharp,dense_slow_epsilon,dense_low_lr --jobs-values 20 --ddt-values 0.5,1.0 --arrival-rates 50,100 --train-instance-seeds 101,202 --validation-instance-seeds 505 --test-instance-seeds 303,404 --training-seeds 11,22,33,44,55,66,77,88,99,110 --episodes 1000 --validation-every 50
```

This run trained 40 DQN checkpoints: 4 reward/hyperparameter variants, 10 independent training seeds per variant, and 1,000 episodes per seed. Each checkpoint was selected by validation tardiness and evaluated only on held-out test instances. All 912 validation/test result rows completed successfully, with 0 failed rows.

Manuscript-facing variant ranking against the primary SPT baseline:

| Rank | Variant | Training runs | DQN mean tardiness | SPT mean tardiness | DQN-SPT | SPT p | W/L/T vs SPT | Errors |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 1 | dense | 10 | 0.267804 | 0.273815 | -0.006011 | 0.002808 | 35/18/27 | 0 |
| 2 | dense_slow_epsilon | 10 | 0.268119 | 0.273815 | -0.005696 | 0.002483 | 32/16/32 | 0 |
| 3 | dense_low_lr | 10 | 0.270419 | 0.273815 | -0.003396 | 0.077089 | 33/21/26 | 0 |
| 4 | sharp | 10 | 0.272242 | 0.273815 | -0.001573 | 0.310247 | 35/27/18 | 0 |

The best variant was the dense tardiness-delta reward with the default learning rate and epsilon decay. Its DQN mean held-out tardiness was `0.267804`, compared with `0.273815` for `SPT_DR_O`. The paired comparison against SPT was significant (`p = 0.002808`) with 35 wins, 18 losses, and 27 ties across 80 paired test comparisons.

ATC-style due-date-aware dispatching is treated as a stress test and future-work target rather than the headline baseline. In the completed run, the dense DQN was statistically indistinguishable from `ATC_DR_O`: DQN mean `0.267804`, ATC mean `0.266695`, paired difference `+0.001109`, and `p = 0.123047`. The correct publication-strength claim is therefore that validation-selected dense-reward DQN significantly improves on SPT in this held-out matrix, while future work should target consistent improvement over ATC-style policies.

## Broad Checkpoint Generalization Study

Command:

```bash
python3 -m djss_rl.cli checkpoint-study --output-dir outputs/dense-broad-generalization-20260720 --checkpoint-glob 'outputs/paper-study-20260719/dense/agents/seed-*/Best_agent*.pth' --jobs-values 20,50,100 --ddt-values 0.5,1.0,1.5 --arrival-rates 50,100,200 --test-instance-seeds 606,707,808,909,1001
```

This evaluated the 10 dense DQN checkpoints from the paper study on 135 additional held-out generated instances without retraining. All 2,565 rows completed successfully: 1,350 DQN checkpoint-instance rows and 1,215 dispatching-rule baseline rows.

Overall ranking:

| Rank | Method | n | Mean tardiness | Std | 95% CI |
|---:|---|---:|---:|---:|---:|
| 1 | SPT_DR_O | 135 | 0.320356 | 0.240501 | 0.040570 |
| 2 | MRT_DR_O | 135 | 0.368362 | 0.239790 | 0.040450 |
| 3 | DQN | 1350 | 0.371964 | 0.277143 | 0.014784 |
| 4 | ATC_DR_O | 135 | 0.375809 | 0.280666 | 0.047345 |
| 5 | LRT_DR_O | 135 | 0.391062 | 0.271928 | 0.045872 |
| 6 | SLK_DR_O | 135 | 0.424921 | 0.304870 | 0.051428 |
| 7 | EDD_DR_O | 135 | 0.428607 | 0.307630 | 0.051894 |
| 8 | MCR_DR_O | 135 | 0.433106 | 0.312882 | 0.052780 |
| 9 | LSPO_DR_O | 135 | 0.433695 | 0.308948 | 0.052116 |
| 10 | CR_DR_O | 135 | 0.448320 | 0.305253 | 0.051493 |

Key paired comparisons:

| Baseline | Mean DQN-baseline difference | Wilcoxon p | Wins | Losses | Ties |
|---|---:|---:|---:|---:|---:|
| SPT_DR_O | +0.051608 | 0.000000 | 226 | 834 | 290 |
| MRT_DR_O | +0.003602 | 0.019782 | 730 | 497 | 123 |
| ATC_DR_O | -0.003845 | 0.000677 | 128 | 93 | 1129 |
| LRT_DR_O | -0.019098 | 0.000000 | 930 | 288 | 132 |
| CR_DR_O | -0.076356 | 0.000000 | 1191 | 13 | 146 |

Per-size breakdown:

| Jobs | SPT mean | ATC mean | DQN mean | Comment |
|---:|---:|---:|---:|---|
| 20 | 0.171043 | 0.168837 | 0.169109 | DQN is competitive with ATC and slightly better than SPT. |
| 50 | 0.299300 | 0.339779 | 0.340441 | SPT is clearly stronger on medium instances. |
| 100 | 0.490726 | 0.618811 | 0.606341 | DQN beats ATC but does not match SPT on larger instances. |

This is a useful external validity result, not a replacement for retraining on the expanded matrix. It shows that the dense DQN checkpoints generalize well enough to beat several weaker rules and slightly improve on ATC overall, but they do not beat SPT across the broader 20/50/100-job generated matrix. The best publishable interpretation is therefore: the paper-study result is valid for its held-out matrix, and the broad checkpoint study identifies scale generalization as the main remaining research gap.

## Expanded Dense Retraining Pilot

Command:

```bash
python3 -m djss_rl.cli paper-study --output-dir outputs/expanded-dense-pilot-20260720 --variants dense --jobs-values 20,50,100 --ddt-values 0.5,1.0,1.5 --arrival-rates 50,100,200 --train-instance-seeds 101 --validation-instance-seeds 505 --test-instance-seeds 707 --training-seeds 66 --episodes 100 --validation-every 25
```

This was a feasibility pilot for the full expanded retraining study: 1 dense-reward DQN run, 100 episodes, 27 training instances, 27 validation instances, and 27 held-out test instances. It completed with 0 failed rows. The selected checkpoint was from episode 74, with best validation tardiness `0.377581`.

Held-out test ranking:

| Rank | Method | n | Mean tardiness | Median | 95% CI |
|---:|---|---:|---:|---:|---:|
| 1 | SPT_DR_O | 27 | 0.315177 | 0.269521 | 0.094035 |
| 2 | MRT_DR_O | 27 | 0.359163 | 0.395466 | 0.091829 |
| 3 | ATC_DR_O | 27 | 0.364898 | 0.404282 | 0.108853 |
| 4 | DQN | 27 | 0.364898 | 0.404282 | 0.108853 |

The pilot DQN matched `ATC_DR_O` exactly on this held-out split and did not beat `SPT_DR_O`: DQN-SPT mean difference `+0.049722`, Wilcoxon `p = 0.003249`, with 7 wins, 17 losses, and 3 ties. It did beat weaker rules including `CR_DR_O`, `MCR_DR_O`, `LSPO_DR_O`, `EDD_DR_O`, `SLK_DR_O`, and `LRT_DR_O`.

Policy trace command:

```bash
python3 -m djss_rl.cli trace-policy --output-dir outputs/expanded-dense-pilot-20260720/policy-trace --checkpoint 'outputs/expanded-dense-pilot-20260720/dense/agents/seed-66/Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth' --dataset-glob 'outputs/expanded-dense-pilot-20260720/dense/test/datasets/*.ini'
```

The policy trace confirms why DQN and `ATC_DR_O` are identical in the pilot: the selected DQN checkpoint chose `ATC_DR_O` for all 12,314 dispatching decisions across all 27 held-out instances. No other action was selected.

Interpretation: expanded retraining is technically feasible and validation improved during training, but 100 episodes and 1 seed are too small for a publishable expanded-matrix claim. The next serious run should keep the expanded matrix, increase training budget and independent seeds, and add policy-behavior diagnostics so the learned agent cannot be mistaken for a single-rule clone.

## Interpretation

The expanded baseline matrix shows that `SPT_DR_O` is the strongest broad default baseline across the larger generated-instance grid. The DQN studies are now more than smoke tests: the 10-seed dense paper study shows that validation-based checkpointing and dense reward shaping can produce a DQN policy that significantly beats SPT on the held-out RL matrix.

The main limitation is no longer just ATC; it is broader scale generalization. The best DQN result beats SPT on the original held-out RL matrix, while the wider checkpoint-only study and the 100-episode expanded retraining pilot both show that SPT remains stronger across larger 50- and 100-job cases. A careful paper can claim a reproducible RL pipeline, a competitive dense-reward DQN variant, and statistically significant improvement over SPT on the selected held-out matrix. A stronger paper should retrain longer on the expanded matrix with multiple seeds, add standard benchmark-derived instances where possible, and report ATC-style rules as a stress-test family rather than hiding them.
