# Project Review

## What This Project Does

The project implements a reinforcement-learning approach to dynamic job shop scheduling. A custom Gymnasium-like environment simulates jobs, operations, work centers, machines, stochastic job arrivals, machine degradation, and optional preventive/corrective maintenance. The objective is to reduce tardiness while tracking energy and maintenance-related costs.

The agent is a PyTorch Dueling DQN with target-network updates and prioritized replay. The action space selects dispatching rules such as MRT, LRT, SPT, EDD, MCR, CR, ATC, LSPO, and SLK. The project also includes baseline evaluation helpers, W&B experiment logging, Optuna hyperparameter scaffolding, one checkpoint, and archived W&B run output.

## Improvements Already Applied

- Stripped notebook execution outputs and bulky widget metadata.
- Removed the commented W&B API key line from the notebook.
- Changed W&B to offline mode by default.
- Disabled W&B smoke tests, evaluation, and training by default behind environment variables.
- Replaced notebook `pip install` cells with a `requirements.txt` workflow.
- Preserved the original dependency freeze in UTF-8 as `requirements-freeze.txt`.
- Fixed the maintenance action index check so maintenance uses the extra action instead of colliding with a dispatching rule.
- Fixed the total-cost calculation to include tardiness, energy, and maintenance cost instead of double-counting maintenance.
- Fixed the makespan call in evaluation.
- Let `train_agents` accept `episodes` and `eval_every`.
- Stopped Optuna from overwriting suggested hyperparameters with fixed values.
- Added `.gitignore` rules for generated notebook, W&B, cache, and environment files.

## Recommended Next Improvements

1. Replace or regenerate `Dataset 50_0.5_0.02.ini`.
2. Extract the notebook into modules:
   - `environment.py`
   - `scenario.py`
   - `agent.py`
   - `training.py`
   - `evaluation.py`
3. Add smoke tests for environment reset/step behavior and action legality.
4. Add a deterministic dataset loader so validation does not rely on newly generated random worlds.
5. Add a CLI for training and evaluation instead of relying on notebook cell order.
6. Add a short experiment report comparing the trained checkpoint with dispatching-rule baselines.
7. Consider using W&B artifacts or GitHub releases for model checkpoints rather than keeping large experiment outputs in the repo.
