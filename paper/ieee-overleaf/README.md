# IEEE Overleaf Manuscript

This folder is a self-contained Overleaf project for:

> When Deep Q-Network Scheduling Becomes a Dispatching Rule: A Reproducible Diagnostic Study of Dynamic Job Shops

## Before Submission

Replace the author metadata near the top of `main.tex`:

- author name;
- department and institution;
- city and country;
- corresponding email;
- funding statement, if applicable.

The manuscript intentionally makes a diagnostic and reproducibility claim. It does not claim that DQN outperforms the strongest dispatching rule.

## Use in Overleaf

1. Upload the ZIP containing this folder to Overleaf as a new project.
2. Keep `main.tex` as the main document.
3. Select pdfLaTeX as the compiler.
4. Compile with BibTeX enabled; Overleaf handles the required reruns automatically.

The project uses the standard `IEEEtran` class supplied by Overleaf.

## Regenerate Figures

From the repository root:

```bash
MPLCONFIGDIR=/private/tmp/djss-matplotlib \
python3 paper/ieee-overleaf/scripts/make_figures.py
```

The script reads the committed result CSVs and rewrites the PDF and PNG figures under `figures/`.

## Evidence Sources

- `outputs/expanded-dense-multiseed-20260721/dense/rl_results.csv`
- `outputs/expanded-dense-multiseed-20260721/dense/policy-trace/checkpoint_policy_trace.csv`
- `outputs/or-library-benchmark-study-20260723-clean/benchmark_results.csv`
- `outputs/or-library-benchmark-study-20260723-clean/policy-trace/checkpoint_policy_trace.csv`

All numeric claims in the manuscript are derived from these artifacts.
