# Seed Study Results

A three-seed replication study was run to check whether the full-training result was robust to training randomness.

Each seed used:

```bash
python3 -m djss_rl.cli train --episodes 1000 --seed <seed> --output-dir outputs/seed-study-20260717/seed-<seed>
```

Each saved best checkpoint was then evaluated on the restored `Dataset 50_0.5_0.02.ini` dataset.

## Baselines

| Method | Tardiness rate | Makespan | Mean machine utilization |
|---|---:|---:|---:|
| SPT_DR_O | 0.6207 | 45h 54m | 75.09% |
| MRT_DR_O | 0.6355 | 42h 14m | 95.31% |
| Original checkpoint | 0.6404 | 50h 55m | 79.44% |

Lower tardiness rate is better.

## Seeded Training Runs

| Seed | Best training episode | Best training reward | Best episode tardiness | Evaluation tardiness | Evaluation makespan | Evaluation utilization |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 652 | -4.00 | 0.60 | 0.5862 | 40h 51m | 93.79% |
| 202 | 835 | 4.50 | 0.59 | 0.6232 | 50h 29m | 76.84% |
| 303 | 127 | 1.75 | 0.59 | 0.5887 | 40h 51m | 96.25% |

## Aggregate

- Mean evaluation tardiness: 0.5993
- Standard deviation of evaluation tardiness: 0.0207
- Best evaluation tardiness: 0.5862, seed 101
- Worst evaluation tardiness: 0.6232, seed 202
- Mean evaluation makespan: 44h 04m
- Mean evaluation utilization: 88.96%
- Seeds beating `SPT_DR_O` on tardiness: 2 / 3
- Seeds beating the original checkpoint on tardiness: 3 / 3

## Interpretation

The seed study supports the claim that retraining improves the project, but it also shows meaningful variance.

The average seeded checkpoint tardiness rate was 0.5993, which is better than the strongest simple baseline, `SPT_DR_O`, at 0.6207. Two of three seeds beat `SPT_DR_O`; all three beat the original checkpoint.

Training reward did not perfectly predict evaluation quality. Seed 202 had the strongest best training reward, 4.50, but its evaluated checkpoint had the weakest seeded tardiness rate, 0.6232. Seed 101 had the weakest best training reward, -4.00, but the best evaluated tardiness rate, 0.5862.

That means the training loop is finding useful policies, but the current reward signal and evaluation objective are not fully aligned. For research reporting, the safest statement is that retraining is promising and usually improves tardiness on this restored dataset, but the method should be reported with multi-seed averages rather than a single best run.
