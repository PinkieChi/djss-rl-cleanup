# Evaluation Results

Evaluation was run with `RUN_DJSS_EVALUATION=1` and `RUN_DJSS_TRAINING` disabled, using the restored `Dataset 50_0.5_0.02.ini`.

## Smoke-Test Context

- Jobs: 50
- Machines: 15
- Operations: 406
- Observation dimension: 14
- Action dimension: 9
- Training during this run: no
- W&B online sync: no

## Results

| Method | Tardiness rate | Makespan | Mean machine utilization |
|---|---:|---:|---:|
| SPT_DR_O | 0.6207 | 45h 54m | 75.09% |
| MRT_DR_O | 0.6355 | 42h 14m | 95.31% |
| Trained checkpoint | 0.6404 | 50h 55m | 79.44% |
| ATC_DR_O | 0.7167 | 47h 18m | 74.44% |
| LRT_DR_O | 0.7414 | 53h 59m | 75.93% |
| SLK_DR_O | 0.8177 | 55h 23m | 72.71% |
| LSPO_DR_O | 0.8276 | 55h 20m | 72.73% |
| EDD_DR_O | 0.8374 | 51h 38m | 78.43% |
| MCR_DR_O | 0.8424 | 53h 41m | 75.82% |
| CR_DR_O | 0.8547 | 52h 58m | 74.95% |

Lower tardiness rate is better. On this restored dataset, the trained checkpoint ranked behind `SPT_DR_O` and close to `MRT_DR_O`, which suggests the checkpoint should be treated as a partial or unvalidated model rather than a final best-performing policy.

## Extracted CLI Verification

After extracting the notebook implementation into `djss_rl`, the same safe evaluation was rerun through:

```bash
python -m djss_rl.cli evaluate
```

The extracted-module run reproduced the same tardiness rates and makespans shown above.

A one-episode training smoke run was also verified without overwriting the bundled checkpoint:

```bash
python -m djss_rl.cli train --episodes 1 --output-dir outputs/training
```

That short run completed with `best_score -124.25`. It is only a training-path smoke test, not a meaningful trained model.

For the full 1000-episode training result, see `docs/full_training_results.md`.
