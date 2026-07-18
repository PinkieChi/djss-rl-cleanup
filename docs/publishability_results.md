# Publishability Experiment Results

This document records the larger generated-instance experiments run on July 18, 2026. The raw CSV, logs, generated datasets, and trained checkpoints are stored under `outputs/` locally and are packaged separately for backup.

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

## Interpretation

The expanded baseline matrix shows that `SPT_DR_O` is the strongest broad default baseline in this implementation. The held-out DQN study is encouraging as an engineering milestone because the agent runs reproducibly and generalizes into the neighborhood of strong dispatching rules, but it does not yet support a publication claim that DQN outperforms simple heuristics.

The next publishable step is not another small run of the same model. The research contribution should improve the RL formulation: reward shaping aligned with tardiness reduction, richer state features, validation-based checkpoint selection, more training seeds, and comparisons on larger held-out matrices that include standard benchmark families where possible.
