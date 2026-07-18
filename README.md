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

Run a generated-instance baseline experiment matrix:

```bash
python -m djss_rl.cli experiment --jobs-values 20 --ddt-values 0.5,1.0,1.5 --arrival-rates 50,100,200 --seeds 101,202,303
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
- Publication-strength claims need generated or benchmark-derived instance matrices, multi-seed runs, and statistical summaries. The `experiment` command now provides the first baseline matrix workflow.
