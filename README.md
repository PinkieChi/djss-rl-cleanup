# DQN-Based Dynamic Job Shop Scheduling

This project explores dynamic job shop scheduling with a Dueling Double-DQN-style agent. The notebook builds a custom Gymnasium environment with stochastic job arrivals, dispatching-rule actions, optional maintenance planning, PyTorch training, W&B logging, Optuna scaffolding, and baseline evaluation against dispatching heuristics.

## Contents

- `DQN_based_Dynamic_Job_Shop_Scheduling_tardiness.ipynb` - main cleaned notebook.
- `Best_agent_hidden_layers_7neurons_per_layer_[207, 145, 78, 79, 205, 105, 217]_batch_size_32.pth` - saved PyTorch checkpoint included with the original project.
- `Dataset 50_0.5_0.02.ini` - original dataset artifact. It appears to contain only null bytes; see `docs/dataset_integrity.md`.
- `requirements.txt` - minimal runtime dependencies.
- `requirements-freeze.txt` - original full environment freeze converted to UTF-8 for reference.
- `docs/project_review.md` - summary of the project and recommended next improvements.

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
- Baseline and trained-agent evaluation is skipped unless `RUN_DJSS_EVALUATION=1`.
- Full training is skipped unless `RUN_DJSS_TRAINING=1`.

Example:

```bash
WANDB_MODE=offline RUN_DJSS_TRAINING=1 DJSS_EPISODES=100 jupyter notebook
```

## Current Caveats

- The included `.ini` dataset is corrupt or blank, so reproducible validation needs a regenerated or restored dataset.
- The notebook still contains most implementation code inline. A stronger next step is extracting environment, agent, training, and evaluation code into testable Python modules.
- The saved checkpoint may be from a partial training run based on the archived W&B logs.
