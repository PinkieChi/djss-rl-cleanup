# Publication Statement

This repository is ready to publish as a reproducibility and diagnostic package for DQN-based dynamic job shop scheduling.

## Recommended Claim

The defensible claim is:

> A legacy DQN-based dynamic job shop scheduling notebook was restored, modularized, tested, and extended into a reproducible experiment package. Dense reward shaping and validation-based checkpoint selection produced a DQN variant that significantly improved over the primary SPT baseline on a selected held-out matrix, while broader generalization experiments showed that SPT remains stronger across larger operating regimes.

## What This Package Supports

- Safe notebook execution with expensive work disabled by default.
- Importable Python modules for environment construction, training, evaluation, generated datasets, and experiment summaries.
- Restored dataset and checkpoint evaluation.
- Generated train/validation/test studies with independent training seeds.
- Paired dispatching-rule comparisons with Wilcoxon tests, win/loss/tie counts, confidence intervals, and regime breakdowns.
- Broad checkpoint generalization over 20-, 50-, and 100-job generated instances.
- Policy-trace diagnostics showing when the DQN collapses to a single dispatching rule.

## What This Package Does Not Claim

- It does not prove that DQN is broadly superior to classical dispatching rules.
- It does not prove robust scale generalization across all dynamic job shop regimes.
- It does not prove superiority over ATC-style due-date-aware dispatching.
- It does not provide real-factory or digital-twin validation.

## Publication Route

The current state is suitable for:

- a thesis chapter on reproducibility, restoration, and diagnostic evaluation;
- an open research artifact linked to a paper or appendix;
- a workshop or preprint focused on lessons learned from DQN rule-selection in DJSS;
- background evidence motivating stronger designs such as regime-aware or Pareto-conditioned PPO.

For a full journal submission claiming algorithmic superiority, the project still needs expanded-matrix retraining with multiple seeds, benchmark-derived instances, and transparent comparison against the strongest baselines.

Before archival release with a DOI, choose and add a software license. Until a license is declared, the repository can be read and cited, but reuse rights are ambiguous.

## Suggested Title

Reproducible Evaluation of DQN-Based Dispatching-Rule Selection for Dynamic Job Shop Scheduling

## Suggested Abstract

Dynamic job shop scheduling is often used to evaluate reinforcement learning methods, but legacy notebook implementations can be difficult to reproduce and may overstate learned-policy performance against strong dispatching rules. This artifact restores and modularizes a DQN-based dynamic job shop scheduling project, adds safe execution defaults, generated-instance train/validation/test protocols, paired statistical summaries, and policy-trace diagnostics. In a 10-seed dense-reward paper study, validation-selected DQN improved over the primary SPT baseline on the selected held-out matrix. However, broad checkpoint generalization over 20-, 50-, and 100-job instances showed that SPT remained stronger overall, and an expanded retraining pilot revealed that one selected DQN checkpoint chose ATC for every held-out dispatching decision. These results support a cautious conclusion: validation and dense reward shaping can improve DQN rule selection in restricted settings, but stronger RL designs are required before claiming broad superiority over classical dispatching rules.
