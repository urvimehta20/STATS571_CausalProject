from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.cdnots.core import CDNOTS, CDNOTSConfig
from src.cdnots.metrics import precision_recall_f1, shd


def simulate_from_dag(n_nodes: int, n_obs: int, rng: np.random.Generator):
    dag = nx.gn_graph(n_nodes, seed=int(rng.integers(0, 1_000_000))).reverse()
    order = list(nx.topological_sort(dag))
    x = np.zeros((n_obs, n_nodes))
    noise = rng.normal(size=(n_obs, n_nodes))
    for j in order:
        parents = list(dag.predecessors(j))
        if parents:
            beta = rng.uniform(-0.8, 0.8, size=len(parents))
            x[:, j] = np.tanh(x[:, parents] @ beta) + noise[:, j]
        else:
            x[:, j] = noise[:, j]
    cols = [f"X{i}" for i in range(n_nodes)]
    return pd.DataFrame(x, columns=cols), dag


def run(quick: bool = False):
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    node_grid = [3, 4, 5, 6, 8, 10, 15] if not quick else [3, 5]
    obs_grid = [50, 150, 300, 500, 1000] if not quick else [50, 150]
    ci_methods = ["parcorr", "kcit_hbe", "rcot_hbe", "cmiknn"]
    n_graphs = 50 if not quick else 5
    records = []
    for n_nodes in node_grid:
        for n_obs in obs_grid:
            for graph_id in tqdm(range(n_graphs), desc=f"nodes={n_nodes},obs={n_obs}"):
                df, dag = simulate_from_dag(n_nodes, n_obs, rng)
                true_edges = set(dag.edges())
                for method in ci_methods:
                    cfg = CDNOTSConfig(max_lag=1, ci_method=method, alpha=0.05, max_condition_set=2)
                    model = CDNOTS(cfg)
                    start = time.perf_counter()
                    res = model.fit(df)
                    elapsed = time.perf_counter() - start
                    pred = {(int(u.split("X")[-1].split("_")[0]), int(v.split("X")[-1].split("_")[0]))
                            for u, v in res["graph"].edges()
                            if u.startswith("X") and v.startswith("X") and "_t-" not in u and "_t-" not in v}
                    m = precision_recall_f1(pred, true_edges)
                    records.append(
                        {
                            "n_nodes": n_nodes,
                            "n_obs": n_obs,
                            "graph_id": graph_id,
                            "ci_method": method,
                            "precision": m["precision"],
                            "recall": m["recall"],
                            "f1": m["f1"],
                            "shd": shd(pred, true_edges),
                            "runtime_sec": elapsed,
                        }
                    )
    out = pd.DataFrame(records)
    out.to_csv(out_dir / "simulation_metrics.csv", index=False)
    summary = (
        out.groupby(["ci_method", "n_nodes", "n_obs"], as_index=False)[["f1", "runtime_sec"]]
        .mean()
        .sort_values(["n_obs", "n_nodes"])
    )
    summary.to_csv(out_dir / "simulation_summary.csv", index=False)
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=summary, x="n_nodes", y="f1", hue="ci_method", style="n_obs", markers=True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / "simulation_f1.png", dpi=150)

    with open(out_dir / "simulation_config.json", "w", encoding="utf-8") as f:
        json.dump({"seed": 0, "node_grid": node_grid, "obs_grid": obs_grid, "n_graphs_each": n_graphs}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a lightweight sanity run.")
    args = parser.parse_args()
    run(quick=args.quick)

