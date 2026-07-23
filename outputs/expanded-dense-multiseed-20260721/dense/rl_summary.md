# RL Generalization Study Summary

Training runs: 3
Test result rows: 972
Failed result rows: 0

## Rows by Split

| Split | Rows |
|---|---:|
| test | 972 |
| validation | 648 |

## Training Runs

| Training seed | Episodes | Best training score | Best validation tardiness | Selection metric | Checkpoint |
|---:|---:|---:|---:|---|---|
| 11 | 500 | 5.571127877709631 | 0.3171586636510609 | validation_tardiness | /Users/c.ngwu.1/Documents/New project/djss-rl-cleanup/outputs/expanded-dense-multiseed-20260721/dense/agents/seed-11/Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth |
| 22 | 500 | 5.656214357585302 | 0.3171586636510609 | validation_tardiness | /Users/c.ngwu.1/Documents/New project/djss-rl-cleanup/outputs/expanded-dense-multiseed-20260721/dense/agents/seed-22/Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth |
| 33 | 500 | 5.6510288171094185 | 0.37187739918731216 | validation_tardiness | /Users/c.ngwu.1/Documents/New project/djss-rl-cleanup/outputs/expanded-dense-multiseed-20260721/dense/agents/seed-33/Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth |

## DQN by Training Seed

| Training seed | Held-out instances | Mean tardiness | Median | Std | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 11 | 81 | 0.323933 | 0.333742 | 0.242185 | 0.052743 |
| 22 | 81 | 0.323933 | 0.333742 | 0.242185 | 0.052743 |
| 33 | 81 | 0.372714 | 0.408589 | 0.240557 | 0.052388 |

## Test Ranking

| Rank | Method | n | Mean tardiness | Median | IQR | Std | 95% CI | Mean makespan | Mean utilization |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | SPT_DR_O | 81 | 0.323933 | 0.333742 | 0.450000 | 0.242185 | 0.052743 | 4355.074074 | 59.515053% |
| 2 | DQN | 243 | 0.340194 | 0.358247 | 0.457065 | 0.241744 | 0.030395 | 4395.543210 | 60.995314% |
| 3 | MRT_DR_O | 81 | 0.372714 | 0.408589 | 0.428884 | 0.240557 | 0.052388 | 4476.481481 | 63.955836% |
| 4 | ATC_DR_O | 81 | 0.378592 | 0.419598 | 0.531365 | 0.281446 | 0.061293 | 4357.185185 | 60.786035% |
| 5 | LRT_DR_O | 81 | 0.396356 | 0.428221 | 0.497530 | 0.275055 | 0.059901 | 4625.148148 | 60.488037% |
| 6 | SLK_DR_O | 81 | 0.429171 | 0.466667 | 0.617733 | 0.307627 | 0.066994 | 4525.000000 | 62.478368% |
| 7 | EDD_DR_O | 81 | 0.432264 | 0.473684 | 0.616110 | 0.310017 | 0.067515 | 4597.370370 | 61.032255% |
| 8 | LSPO_DR_O | 81 | 0.437534 | 0.476074 | 0.630848 | 0.311105 | 0.067752 | 4581.197531 | 61.627495% |
| 9 | MCR_DR_O | 81 | 0.437556 | 0.480000 | 0.634608 | 0.314985 | 0.068597 | 4593.901235 | 61.253950% |
| 10 | CR_DR_O | 81 | 0.451950 | 0.497481 | 0.589153 | 0.305989 | 0.066638 | 4495.222222 | 61.807007% |

## DQN Against Baselines

Negative differences mean DQN had lower tardiness than the baseline.

| Baseline | Compared pairs | Mean difference | Median difference | Wilcoxon p | Rank-biserial r | Wins | Losses | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CR_DR_O | 243 | -0.111756 | -0.082902 | 0.000000 | -0.966957 | 202 | 17 | 24 |
| MCR_DR_O | 243 | -0.097363 | -0.062577 | 0.000000 | -0.860291 | 177 | 37 | 29 |
| LSPO_DR_O | 243 | -0.097340 | -0.070352 | 0.000000 | -0.873798 | 180 | 30 | 33 |
| EDD_DR_O | 243 | -0.092070 | -0.062500 | 0.000000 | -0.882717 | 181 | 32 | 30 |
| SLK_DR_O | 243 | -0.088978 | -0.059585 | 0.000000 | -0.834973 | 165 | 46 | 32 |
| LRT_DR_O | 243 | -0.056162 | -0.056250 | 0.000000 | -0.880156 | 185 | 29 | 29 |
| ATC_DR_O | 243 | -0.038398 | -0.013158 | 0.000000 | -0.542100 | 141 | 72 | 30 |
| MRT_DR_O | 243 | -0.032520 | -0.006250 | 0.000000 | -0.933333 | 122 | 18 | 103 |
| SPT_DR_O | 243 | 0.016260 | 0.000000 | 0.000000 | 0.931992 | 9 | 61 | 173 |

## Regime Breakdown

### By Jobs

| Jobs | Method | n | Mean tardiness | Median | 95% CI |
|---:|---|---:|---:|---:|---:|
| 20 | SPT_DR_O | 27 | 0.174850 | 0.040000 | 0.083922 |
| 20 | DQN | 81 | 0.176998 | 0.040000 | 0.047671 |
| 20 | MRT_DR_O | 27 | 0.181295 | 0.040000 | 0.082989 |
| 20 | ATC_DR_O | 27 | 0.171581 | 0.026667 | 0.084496 |
| 50 | SPT_DR_O | 27 | 0.306143 | 0.241206 | 0.081087 |
| 50 | DQN | 81 | 0.332023 | 0.266839 | 0.045348 |
| 50 | MRT_DR_O | 27 | 0.383782 | 0.367876 | 0.072375 |
| 50 | ATC_DR_O | 27 | 0.345480 | 0.298969 | 0.086532 |
| 100 | SPT_DR_O | 27 | 0.490807 | 0.433824 | 0.067932 |
| 100 | DQN | 81 | 0.511559 | 0.511656 | 0.036919 |
| 100 | MRT_DR_O | 27 | 0.553064 | 0.561713 | 0.054504 |
| 100 | ATC_DR_O | 27 | 0.618715 | 0.668766 | 0.070976 |

### By DDT

| DDT | Method | n | Mean tardiness | Median | 95% CI |
|---:|---|---:|---:|---:|---:|
| 0.5 | SPT_DR_O | 27 | 0.587419 | 0.579897 | 0.037506 |
| 0.5 | DQN | 81 | 0.591988 | 0.592965 | 0.021533 |
| 0.5 | MRT_DR_O | 27 | 0.601126 | 0.616580 | 0.038056 |
| 0.5 | ATC_DR_O | 27 | 0.625932 | 0.595477 | 0.052786 |
| 1.0 | SPT_DR_O | 27 | 0.243996 | 0.241206 | 0.069942 |
| 1.0 | DQN | 81 | 0.264661 | 0.253886 | 0.042011 |
| 1.0 | MRT_DR_O | 27 | 0.305992 | 0.326633 | 0.078291 |
| 1.0 | ATC_DR_O | 27 | 0.300241 | 0.258794 | 0.095564 |
| 1.5 | SPT_DR_O | 27 | 0.140386 | 0.090674 | 0.056566 |
| 1.5 | DQN | 81 | 0.163932 | 0.100503 | 0.036678 |
| 1.5 | MRT_DR_O | 27 | 0.211024 | 0.173367 | 0.074343 |
| 1.5 | ATC_DR_O | 27 | 0.209602 | 0.118090 | 0.092702 |

### By Arrival rate

| Arrival rate | Method | n | Mean tardiness | Median | 95% CI |
|---:|---|---:|---:|---:|---:|
| 50 | SPT_DR_O | 27 | 0.387410 | 0.433824 | 0.096327 |
| 50 | DQN | 81 | 0.405279 | 0.469939 | 0.054300 |
| 50 | MRT_DR_O | 27 | 0.441015 | 0.512500 | 0.091522 |
| 50 | ATC_DR_O | 27 | 0.481754 | 0.513333 | 0.117887 |
| 100 | SPT_DR_O | 27 | 0.309397 | 0.285539 | 0.089356 |
| 100 | DQN | 81 | 0.327196 | 0.326633 | 0.051529 |
| 100 | MRT_DR_O | 27 | 0.362794 | 0.404412 | 0.090885 |
| 100 | ATC_DR_O | 27 | 0.358372 | 0.458438 | 0.100230 |
| 200 | SPT_DR_O | 27 | 0.274993 | 0.231902 | 0.086222 |
| 200 | DQN | 81 | 0.288106 | 0.265464 | 0.049383 |
| 200 | MRT_DR_O | 27 | 0.314332 | 0.314110 | 0.086545 |
| 200 | ATC_DR_O | 27 | 0.295649 | 0.319899 | 0.089919 |

## Interpretation Guardrail

Only held-out test rows are used for the ranking and paired comparisons. Validation rows are available in the raw CSV for checkpoint-selection auditing.
