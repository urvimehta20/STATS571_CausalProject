# discovery2: causal-learn CD-NOD

This folder runs `causal-learn` CD-NOD directly on datasets in `data/raw/`.

## Inputs

- `data/raw/famafrench_apple_daily.csv`
- `data/raw/macro_countries_monthly.csv`

## Script

- `discovery2/run_causal_learn.py`

## Usage

From repository root:

```bash
python discovery2/run_causal_learn.py --dataset famafrench
python discovery2/run_causal_learn.py --dataset macro --country US
python discovery2/run_causal_learn.py --dataset macro --country all
```

## Outputs (`discovery2/outputs/`)

- `cdnod_<run>_edges.csv` — full endpoint matrix listing
- `cdnod_<run>_directed_edges.csv` — `from` → `to` for fully directed arcs
- `cdnod_<run>_undirected_edges.csv` — ambiguous / undirected pairs
- `cdnod_<run>_nodes.csv`, plus `.png` / `.pdf` / `.dot` when Graphviz is available

## Lecture-style follow-up (e.g. Lecture 13 adjustment)

```bash
python -m experiments.lecture13_graph_adjustment --tag famafrench --z SMB --y HML
python -m experiments.lecture13_graph_adjustment --tag macro_US --z unemployment --y cpi \
  --country US --cpi-diff --extra-controls ppi
```

Uses **parents(Z)** from the directed subgraph as default covariates; add `--extra-controls` when the graph has no parents of Z or you want a larger backdoor set.
