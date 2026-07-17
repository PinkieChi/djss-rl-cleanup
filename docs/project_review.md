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
- Replaced the corrupt null-byte dataset with a valid restored `.ini` dataset.
- Added a deterministic notebook loader for the restored `.ini` dataset.
- Ran safe evaluation on the restored dataset without training or W&B sync.
- Fixed the maintenance action index check so maintenance uses the extra action instead of colliding with a dispatching rule.
- Fixed the total-cost calculation to include tardiness, energy, and maintenance cost instead of double-counting maintenance.
- Fixed the makespan call in evaluation.
- Let `train_agents` accept `episodes` and `eval_every`.
- Stopped Optuna from overwriting suggested hyperparameters with fixed values.
- Added `.gitignore` rules for generated notebook, W&B, cache, and environment files.

## Recommended Next Improvements

1. Re-export the dataset with stable machine IDs or names instead of Python memory-address tokens.
2. Investigate why the trained checkpoint does not beat `SPT_DR_O` on the restored dataset.
3. Extract the notebook into modules:
   - `environment.py`
   - `scenario.py`
   - `agent.py`
   - `training.py`
   - `evaluation.py`
4. Add smoke tests for environment reset/step behavior and action legality.
5. Add a CLI for training and evaluation instead of relying on notebook cell order.
6. Consider using W&B artifacts or GitHub releases for model checkpoints rather than keeping large experiment outputs in the repo.
