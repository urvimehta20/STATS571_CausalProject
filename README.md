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

