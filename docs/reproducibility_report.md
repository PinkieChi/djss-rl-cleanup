# Reproducibility Report

## Artifact Summary

This repository contains a cleaned implementation of a DQN-based dynamic job shop scheduling project. The original notebook has been preserved as a research narrative, while the core environment, agent, training, evaluation, dataset generation, and experiment tooling have been extracted into the `djss_rl` package.

The package is designed to answer a narrow reproducibility question:

> Can the original DQN scheduling project be executed, evaluated, extended to generated held-out instances, and interpreted against strong dispatching-rule baselines?

The answer is yes. The stronger scientific answer is more cautious: the DQN pipeline is reproducible and competitive in selected regimes, but it does not yet establish broad superiority over dispatching rules.

## Main Reproducibility Improvements

- Safe notebook defaults: W&B, training, and expensive evaluation are opt-in.
- Restored valid `.ini` dataset artifact.
- Importable Python modules under `djss_rl/`.
- CLI commands for smoke tests, evaluation, training, generated datasets, JSPLIB conversion, baseline matrices, RL studies, paper studies, checkpoint generalization, and policy tracing.
- Unit tests covering notebook loading, environment construction, baseline scheduling, checkpoint evaluation, generated datasets, JSPLIB conversion, study writers, and policy tracing.
- Markdown and CSV outputs for experiment results.

## Core Commands

Smoke test:

```bash
python3 -m djss_rl.cli smoke
```

Baseline and checkpoint evaluation:

```bash
python3 -m djss_rl.cli evaluate
```

Primary paper study:

```bash
python3 -m djss_rl.cli paper-study \
  --output-dir outputs/paper-study-20260719 \
  --variants dense,sharp,dense_slow_epsilon,dense_low_lr \
  --jobs-values 20 \
  --ddt-values 0.5,1.0 \
  --arrival-rates 50,100 \
  --train-instance-seeds 101,202 \
  --validation-instance-seeds 505 \
  --test-instance-seeds 303,404 \
  --training-seeds 11,22,33,44,55,66,77,88,99,110 \
  --episodes 1000 \
  --validation-every 50
```

Broad checkpoint generalization:

```bash
python3 -m djss_rl.cli checkpoint-study \
  --output-dir outputs/dense-broad-generalization-20260720 \
  --checkpoint-glob 'outputs/paper-study-20260719/dense/agents/seed-*/Best_agent*.pth' \
  --jobs-values 20,50,100 \
  --ddt-values 0.5,1.0,1.5 \
  --arrival-rates 50,100,200 \
  --test-instance-seeds 606,707,808,909,1001
```

Policy trace:

```bash
python3 -m djss_rl.cli trace-policy \
  --output-dir outputs/expanded-dense-pilot-20260720/policy-trace \
  --checkpoint 'outputs/expanded-dense-pilot-20260720/dense/agents/seed-66/Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth' \
  --dataset-glob 'outputs/expanded-dense-pilot-20260720/dense/test/datasets/*.ini'
```

## Key Results

### Baseline Matrix

The large generated baseline matrix evaluated 135 generated instances with 9 dispatching rules. `SPT_DR_O` was the strongest broad baseline with mean tardiness `0.320584`.

### Primary Dense DQN Paper Study

The 10-seed dense-reward variant achieved mean held-out tardiness `0.267804` against `SPT_DR_O` at `0.273815` on the selected 20-job held-out matrix. The paired comparison against SPT was significant with Wilcoxon `p = 0.002808`, 35 wins, 18 losses, and 27 ties.

The same dense DQN was statistically indistinguishable from `ATC_DR_O` on that matrix.

### Broad Generalization

The dense checkpoints were evaluated on 135 additional held-out generated instances spanning 20, 50, and 100 jobs. Overall, `SPT_DR_O` ranked first with mean tardiness `0.320356`, while DQN ranked third with `0.371964`. This shows that the selected DQN checkpoints do not generalize broadly enough to beat SPT at larger scales without retraining.

### Expanded Retraining Pilot

A 100-episode, 1-seed expanded-matrix pilot completed successfully, but DQN did not beat SPT. It matched `ATC_DR_O` exactly with mean tardiness `0.364898`.

The policy trace explains the match: the selected DQN checkpoint chose `ATC_DR_O` for all 12,314 held-out dispatching decisions across 27 test instances.

## Recommended Interpretation

The strongest interpretation is:

> Dense rewards and validation checkpointing improve DQN dispatching-rule selection in a restricted held-out matrix, but broader scale generalization remains unresolved. Policy tracing is essential because a learned DQN can collapse to a single classical dispatching rule.

## Reviewer-Safe Limitations

- The restored source dataset contains reconstructed machine identity because the original file encoded compatible machines as memory-address tokens.
- Most publication evidence uses generated instances rather than external benchmark-derived dynamic instances.
- The broad checkpoint study evaluates existing checkpoints without full retraining on the larger matrix.
- The expanded retraining pilot is intentionally small: 1 seed and 100 episodes.
- Strong baselines such as SPT and ATC remain competitive or superior in important regimes.
- No real-factory, MES, or digital-twin validation is included.

## Recommended Next Experiment

For a stronger journal claim, run the full expanded retraining protocol in `docs/manuscript_protocol.md` with multiple training seeds and include benchmark-derived JSPLIB conversions. Keep ATC-style rules visible as stress-test baselines and include policy-trace diagnostics for every selected checkpoint.

