# Paper Study Summary

Lower tardiness is better. Negative DQN-minus-baseline values mean DQN outperformed that baseline.

| Rank | Variant | Training runs | DQN mean tardiness | SPT mean tardiness | DQN-SPT | SPT p | W/L/T vs SPT | ATC mean tardiness | DQN-ATC | ATC p | Errors |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | dense | 3 | 0.340194 | 0.323933 | 0.016260 | 0.000000 | 9/61/173 | 0.378592 | -0.038398 | 0.000000 | 0 |

## Variant Artifacts

| Variant | Results CSV | Summary |
|---|---|---|
| dense | outputs/expanded-dense-multiseed-20260721/dense/rl_results.csv | outputs/expanded-dense-multiseed-20260721/dense/rl_summary.md |
