# Publishability Experiment Results

This document records the larger generated-instance experiments run on July 18-20, 2026. The raw CSV, logs, generated datasets, and trained checkpoints are stored under `outputs/` locally and are packaged separately for backup.

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

Variant ranking:

| Rank | Variant | Training runs | DQN mean tardiness | SPT mean tardiness | DQN-SPT | SPT p | W/L/T vs SPT | ATC mean tardiness | DQN-ATC | ATC p | Errors |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | dense | 10 | 0.267804 | 0.273815 | -0.006011 | 0.002808 | 35/18/27 | 0.266695 | 0.001109 | 0.123047 | 0 |
| 2 | dense_slow_epsilon | 10 | 0.268119 | 0.273815 | -0.005696 | 0.002483 | 32/16/32 | 0.266695 | 0.001424 | 0.139648 | 0 |
| 3 | dense_low_lr | 10 | 0.270419 | 0.273815 | -0.003396 | 0.077089 | 33/21/26 | 0.266695 | 0.003724 | 0.007407 | 0 |
| 4 | sharp | 10 | 0.272242 | 0.273815 | -0.001573 | 0.310247 | 35/27/18 | 0.266695 | 0.005548 | 0.036303 | 0 |

The best variant was the dense tardiness-delta reward with the default learning rate and epsilon decay. Its DQN mean held-out tardiness was `0.267804`, compared with `0.273815` for `SPT_DR_O`. The paired comparison against SPT was significant (`p = 0.002808`) with 35 wins, 18 losses, and 27 ties across 80 paired test comparisons.

The dense DQN did not significantly outperform `ATC_DR_O`. It was slightly worse on mean tardiness (`+0.001109`) and the paired comparison was not significant (`p = 0.123047`), with most comparisons tied. The correct publication-strength claim is therefore that validation-selected dense-reward DQN is competitive with the strongest dispatching rules and significantly improves on SPT in this held-out matrix, not that it dominates all classical heuristics.

## Interpretation

The expanded baseline matrix shows that `SPT_DR_O` is the strongest broad default baseline across the larger generated-instance grid, while `ATC_DR_O` is the strongest rule on the smaller held-out RL test matrix. The DQN studies are now more than smoke tests: the 10-seed dense paper study shows that validation-based checkpointing and dense reward shaping can produce a DQN policy that significantly beats SPT on the held-out RL matrix.

The main limitation is that the best DQN remains statistically indistinguishable from ATC and was evaluated on a relatively small held-out matrix. A careful paper can claim a reproducible RL pipeline and a competitive DQN variant with statistically significant improvement over SPT on the chosen held-out matrix. A stronger paper should extend the same protocol to larger held-out matrices, add standard benchmark-derived instances where possible, and improve the RL formulation enough to challenge ATC rather than only SPT.
