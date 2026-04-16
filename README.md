# CD-NOTS Reproduction

## Scope

- Implements CD-NOTS stages for nonstationary time series with lagged variables.
- Reproduces:
  - Simulation study across node/sample grids.
  - CD-NOTS vs PCMCI benchmark.
  - Case studies:
    - Fama-French + Apple returns.
    - Macroeconomic CPI/PPI/unemployment across countries.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 pip install -r requirements.txt
```

## Data Download

```bash
source .venv/bin/activate
python scripts/download_famafrench_apple.py
python scripts/download_macro_data.py
```

Expected outputs:
- `data/raw/famafrench_apple_daily.csv`
- `data/raw/macro_countries_monthly.csv`

## Run Experiments

```bash
source .venv/bin/activate
python experiments/run_simulations.py
python experiments/run_benchmark_pcmci.py
python experiments/run_case_famafrench_apple.py
python experiments/run_case_macro_countries.py
```

Generated outputs:
- `results/tables/simulation_metrics.csv`
- `results/tables/benchmark_pcmci.csv`
- `results/tables/case_famafrench_apple_edges.csv`
- `results/tables/case_macro_edges.csv`
- `results/figures/case_famafrench_apple_series.png`

## Notes on Fidelity

- CI test wrappers are unified under `src/cdnots/ci_tests.py`.
- Paper hyperparameters (p-value threshold 0.05 and CI-method naming) are preserved in defaults.
- Stage 4 uses a Meek-style closure approximation for remaining directions.
- Data sourcing uses public endpoints:
  - Kenneth French data library.
  - Yahoo Finance.
  - FRED-compatible country macro series.

## Master Planning Docs

- Deep technical roadmap: `docs/master_plan.md`
- Project narrative and results context: `FINAL_PROJECT_WRITEUP.md`

The master plan explicitly tracks:
- paper intent vs implemented features,
- current approximation boundaries,
- test/validation maturity,
- prioritized remaining implementation work.

## Causal-learn Discovery + Graph-Guided Regression

This repository uses a two-step workflow for causal analysis on real data.

1. **Causal discovery with causal-learn (CD-NOD)**  
   `discovery2/run_causal_learn.py` runs `causallearn.search.ConstraintBased.CDNOD` on:
   - Fama-French + Apple features with a time context (`context_time`), or
   - macro features (`unemployment`, `cpi`, `ppi`) with a country context (`context_country`).

   It exports:
   - `*_edges.csv` (full edge encoding),
   - `*_directed_edges.csv` (fully oriented edges),
   - `*_undirected_edges.csv` (ambiguous/non-oriented pairs),
   - `*_nodes.csv` (node metadata),
   - graph visuals (`.dot`, `.png`, optional `.pdf` when Graphviz is available).

2. **Graph-guided regression (Lecture 13 style)**  
   `experiments/lecture13_graph_adjustment.py` reads `cdnod_<tag>_directed_edges.csv`, sets
   the adjustment set to **parents(Z)** in that directed graph, and estimates:

   `Y ~ Z + L`, where `L = parents(Z)`, using OLS with HAC standard errors.

   The script writes one-row summaries to:
   - `results/tables/lecture13_adjust_<tag>_<z>_<y>.csv`

   If `parents(Z)` is empty, you can add `--extra-controls`.

## Python File Roles

Files needed for the causal-learn + graph-guided workflow:

- `scripts/download_famafrench_apple.py`: Downloads/builds `data/raw/famafrench_apple_daily.csv`.
- `scripts/download_macro_data.py`: Downloads/builds `data/raw/macro_countries_monthly.csv`.
- `discovery2/run_causal_learn.py`: Runs causal-learn CD-NOD and writes graph artifacts (`*_edges.csv`, `*_directed_edges.csv`, `*_undirected_edges.csv`, `*_nodes.csv`, `.dot`, `.png`).
- `experiments/lecture13_graph_adjustment.py`: Reads `*_directed_edges.csv`, sets `L = parents(Z)`, and estimates `Y ~ Z + L` with HAC.
- `experiments/run_effect_smb_hml.py`: Convenience runner for the SMB -> HML graph-guided regression/table.

