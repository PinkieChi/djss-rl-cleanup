# Publication Statement

This repository is ready to publish as a reproducibility and diagnostic artifact for DQN-based dynamic job shop scheduling.

It is not yet ready as a full journal paper claiming that DQN broadly outperforms classical dispatching rules.

## Recommended Claim

The defensible claim is:

> A legacy DQN-based dynamic job shop scheduling notebook was restored, modularized, tested, and extended into a reproducible experiment package. Expanded generated-instance retraining and OR-Library-derived benchmark evaluation show that DQN is competitive with strong dispatching rules, but `SPT_DR_O` remains the strongest overall baseline. Policy-trace diagnostics reveal that selected DQN checkpoints collapse to single-rule behavior, motivating stronger adaptive RL designs such as RA-PPO.

## What This Package Supports

- Safe notebook execution with expensive work disabled by default.
- Importable Python modules for environment construction, training, evaluation, generated datasets, benchmark conversion, and experiment summaries.
- Restored dataset and checkpoint evaluation.
- Generated train/validation/test studies with independent training seeds.
- Expanded-matrix retraining over 20-, 50-, and 100-job generated instances.
- OR-Library-derived benchmark conversion and evaluation.
- Paired dispatching-rule comparisons with Wilcoxon tests, win/loss/tie counts, confidence intervals, and regime breakdowns.
- Transparent reporting against all dispatching rules, including `ATC_DR_O`.
- Policy-trace diagnostics showing when DQN collapses to a single dispatching rule.

## What This Package Does Not Claim

- It does not prove that DQN is broadly superior to classical dispatching rules.
- It does not prove robust scale generalization across all dynamic job shop regimes.
- It does not prove superiority over `SPT_DR_O`.
- It does not prove superiority over ATC-style due-date-aware dispatching.
- It does not provide real-factory or digital-twin validation.

## Latest Evidence

The July 23, 2026 journal-readiness run is summarized in `docs/journal_readiness_results_20260723.md`.

Key results:

- Expanded generated-instance retraining: DQN mean tardiness `0.340194`; `SPT_DR_O` mean tardiness `0.323933`.
- OR-Library-derived benchmark study: DQN mean tardiness `0.524708`; `SPT_DR_O` mean tardiness `0.514833`; `ATC_DR_O` mean tardiness `0.528806`.
- Policy tracing: two selected DQN checkpoints chose `SPT_DR_O` for every decision, and one chose `MRT_DR_O` for every decision.

## Publication Route

The current state is suitable for:

- a thesis chapter on reproducibility, restoration, and diagnostic evaluation;
- an open research artifact linked to a paper or appendix;
- a workshop or preprint focused on lessons learned from DQN rule-selection in DJSS;
- background evidence motivating stronger designs such as RA-PPO.

For a full journal submission claiming algorithmic superiority, the next method should address the exposed limitation directly: learn adaptive, regime-aware policies rather than reproducing a single classical rule.

Before archival release with a DOI, choose and add a software license. Until a license is declared, the repository can be read and cited, but reuse rights are ambiguous.

## Suggested Title

Reproducible Diagnostic Evaluation of DQN-Based Dispatching-Rule Selection for Dynamic Job Shop Scheduling

## Suggested Abstract

Dynamic job shop scheduling is often used to evaluate reinforcement learning methods, but legacy notebook implementations can be difficult to reproduce and may overstate learned-policy performance against strong dispatching rules. This artifact restores and modularizes a DQN-based dynamic job shop scheduling project, adds safe execution defaults, generated-instance train/validation/test protocols, OR-Library-derived benchmark conversion, paired statistical summaries, and policy-trace diagnostics. Expanded generated-instance retraining and benchmark-derived evaluation show that DQN is competitive and improves over several weaker dispatching rules, but `SPT_DR_O` remains the strongest overall baseline. Policy traces reveal that selected DQN checkpoints collapse to deterministic single-rule behavior, primarily `SPT_DR_O` or `MRT_DR_O`. These results support a cautious conclusion: DQN rule selection is reproducible and diagnostically useful, but stronger adaptive RL designs are required before claiming broad superiority over classical dispatching rules.
