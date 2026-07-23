# DQN-Based Dynamic Job Shop Scheduling

This project explores dynamic job shop scheduling with a Dueling Double-DQN-style agent. The notebook builds a custom Gymnasium environment with stochastic job arrivals, dispatching-rule actions, optional maintenance planning, PyTorch training, W&B logging, Optuna scaffolding, and baseline evaluation against dispatching heuristics.

## Contents

- `DQN_based_Dynamic_Job_Shop_Scheduling_tardiness.ipynb` - main cleaned notebook.
- `Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth` - saved PyTorch checkpoint included with the original project.
- `Dataset 50_0.5_0.02.ini` - restored valid dataset artifact; see `docs/dataset_integrity.md`.
- `djss_rl/` - extracted Python package for environment construction, agent code, training, and evaluation.
- `tests/` - smoke tests for notebook and package execution paths.
- `requirements.txt` - minimal runtime dependencies.
- `requirements-freeze.txt` - original full environment freeze converted to UTF-8 for reference.
- `docs/project_review.md` - summary of the project and recommended next improvements.
- `docs/evaluation_results.md` - latest safe evaluation results on the restored dataset.
- `docs/publishability_results.md` - generated-instance baseline and held-out DQN study results.
- `docs/journal_readiness_results_20260723.md` - latest expanded-matrix, benchmark-derived, and policy-trace results.
- `docs/manuscript_protocol.md` - next publication-grade experiment protocol and commands.
- `docs/publication_statement.md` - recommended publication claim, boundaries, and abstract.
- `docs/reproducibility_report.md` - manuscript-style artifact report and key results.
- `CITATION.cff` - citation metadata for software/artifact publication.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Safe Notebook Defaults

The notebook has been cleaned so expensive or networked actions are opt-in:

- W&B defaults to offline mode. Set `WANDB_MODE=online` only when you want to sync runs.
- The W&B smoke test is skipped unless `RUN_WANDB_SMOKE_TEST=1`.
- The restored dataset is loaded by default from `Dataset 50_0.5_0.02.ini`; override with `DJSS_DATASET_PATH`.
- Baseline and trained-agent evaluation is skipped unless `RUN_DJSS_EVALUATION=1`.
- Full training is skipped unless `RUN_DJSS_TRAINING=1`.

Example:

```bash
WANDB_MODE=offline RUN_DJSS_TRAINING=1 DJSS_EPISODES=100 jupyter notebook
```

## Terminal Commands

Run a safe smoke test:

```bash
python -m djss_rl.cli smoke
```

Run baseline and checkpoint evaluation without training:

```bash
python -m djss_rl.cli evaluate
```

Generate a stable-ID synthetic dataset:

```bash
python -m djss_rl.cli generate-dataset --output outputs/generated/example.ini --jobs 20 --seed 101
```

Convert a JSPLIB-style benchmark file into the project `.ini` format:

```bash
python -m djss_rl.cli convert-jsplib --input benchmarks/ft06.txt --output outputs/benchmarks/ft06_dynamic.ini --ddt 1.0 --arrival-rate 50 --initial-job-fraction 1.0
```

Run a generated-instance baseline experiment matrix:

```bash
python -m djss_rl.cli experiment --jobs-values 20 --ddt-values 0.5,1.0,1.5 --arrival-rates 50,100,200 --seeds 101,202,303
```

Run a generated-instance DQN generalization study:

```bash
python -m djss_rl.cli rl-study --jobs-values 20 --ddt-values 0.5,1.0 --arrival-rates 50,100 --train-instance-seeds 101,202 --validation-instance-seeds 505 --test-instance-seeds 303,404 --training-seeds 11,22,33 --episodes 1000 --validation-every 50
```

Run a resumable multi-variant paper study:

```bash
python -m djss_rl.cli paper-study --variants dense,sharp,dense_slow_epsilon,dense_low_lr --jobs-values 20 --ddt-values 0.5,1.0 --arrival-rates 50,100 --train-instance-seeds 101,202 --validation-instance-seeds 505 --test-instance-seeds 303,404 --training-seeds 11,22,33,44,55,66,77,88,99,110 --episodes 1000 --validation-every 50
```

Evaluate existing trained checkpoints on a broader held-out matrix:

```bash
python -m djss_rl.cli checkpoint-study --checkpoint-glob 'outputs/paper-study-20260719/dense/agents/seed-*/Best_agent*.pth' --jobs-values 20,50,100 --ddt-values 0.5,1.0,1.5 --arrival-rates 50,100,200 --test-instance-seeds 606,707,808,909,1001
```

Trace which dispatching-rule actions a saved DQN checkpoint selects:

```bash
python -m djss_rl.cli trace-policy --checkpoint 'outputs/expanded-dense-pilot-20260720/dense/agents/seed-66/Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth' --dataset-glob 'outputs/expanded-dense-pilot-20260720/dense/test/datasets/*.ini'
```

Prepare OR-Library benchmark-derived dynamic datasets:

```bash
python scripts/prepare_or_library_benchmarks.py --source benchmarks/or-library/raw/jobshop1.txt --output-dir outputs/or-library-benchmark-derived-20260721/datasets
```

Evaluate baselines and trained checkpoints on benchmark-derived datasets:

```bash
python -m djss_rl.cli benchmark-study --output-dir outputs/or-library-benchmark-study-20260723-clean --dataset-glob 'outputs/or-library-benchmark-derived-20260721/datasets/*.ini' --checkpoint-glob 'outputs/expanded-dense-multiseed-20260721/dense/agents/seed-*/Best_agent*.pth'
```

Trace multiple trained checkpoints:

```bash
python -m djss_rl.cli trace-checkpoints --output-dir outputs/or-library-benchmark-study-20260723-clean/policy-trace --checkpoint-glob 'outputs/expanded-dense-multiseed-20260721/dense/agents/seed-*/Best_agent*.pth' --dataset-glob 'outputs/or-library-benchmark-derived-20260721/datasets/*.ini'
```

Run one training episode from the restored dataset:

```bash
python -m djss_rl.cli train --episodes 1 --seed 101 --output-dir outputs/training
```

Run the smoke test through `unittest` discovery:

```bash
python -m unittest discover -s tests -v
```

## Current Caveats

- The restored `.ini` dataset is loadable by the notebook. Exact original machine identity is limited because the source `.ini` stored compatible machines as Python memory-address tokens rather than machine IDs.
- The project now exposes environment, agent, training, and evaluation code through the `djss_rl` package. The notebook remains as the research narrative and compatibility reference.
- The saved checkpoint evaluated successfully, but it did not beat the strongest simple dispatching baseline on the restored dataset.
- The larger generated-instance baseline matrix found `SPT_DR_O` to be the strongest broad baseline in this implementation.
- Expanded-matrix dense retraining with 3 seeds and 500 episodes completed successfully, but DQN ranked second behind `SPT_DR_O` on held-out generated instances.
- OR-Library-derived benchmark evaluation completed on 60 converted dynamic datasets. DQN again ranked second behind `SPT_DR_O` and slightly ahead of `ATC_DR_O`; see `docs/journal_readiness_results_20260723.md`.
- Policy tracing showed the expanded DQN checkpoints collapsed to single-rule behavior: seeds 11 and 22 selected `SPT_DR_O` for every decision, while seed 33 selected `MRT_DR_O` for every decision.
- Publication-strength claims should be framed around reproducibility, transparent diagnostics, and motivation for stronger adaptive RL designs such as RA-PPO, not broad DQN superiority.

## Publication Status

This repository is ready to publish as a code and reproducibility artifact. Use the framing in `docs/publication_statement.md` and `docs/journal_readiness_results_20260723.md`: the project supports a cautious reproducibility and diagnostic claim, not a broad claim that DQN is superior to dispatching rules.
