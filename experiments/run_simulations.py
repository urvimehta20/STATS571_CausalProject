from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.cdnots.core import CDNOTS, CDNOTSConfig
from src.cdnots.metrics import precision_recall_f1, shd
from src.cdnots.project_io import ProjectPaths, get_logger

LOGGER = get_logger("experiments.run_simulations")


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


@dataclass(frozen=True)
class SimulationConfig:
    seed: int
    node_grid: list[int]
    obs_grid: list[int]
    ci_methods: list[str]
    n_graphs: int


class SimulationExperiment:
    """Application service to run simulation benchmarking end-to-end."""

    def __init__(self, paths: ProjectPaths, config: SimulationConfig):
        self.paths = paths
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    @staticmethod
    def _extract_predicted_edges(edges: Iterable[tuple[str, str]]) -> set[tuple[int, int]]:
        parsed_edges: set[tuple[int, int]] = set()
        for source, target in edges:
            if not (source.startswith("X") and target.startswith("X")):
                continue
            if "_t-" in source or "_t-" in target:
                continue
            source_idx = int(source.split("X")[-1].split("_")[0])
            target_idx = int(target.split("X")[-1].split("_")[0])
            parsed_edges.add((source_idx, target_idx))
        return parsed_edges

    def run(self) -> None:
        self.paths.ensure_standard_dirs()
        records = []
        for n_nodes in self.config.node_grid:
            for n_obs in self.config.obs_grid:
                for graph_id in tqdm(range(self.config.n_graphs), desc=f"nodes={n_nodes},obs={n_obs}"):
                    frame, dag = simulate_from_dag(n_nodes, n_obs, self.rng)
                    true_edges = set(dag.edges())
                    for method in self.config.ci_methods:
                        model = CDNOTS(
                            CDNOTSConfig(max_lag=1, ci_method=method, alpha=0.05, max_condition_set=2)
                        )
                        start = time.perf_counter()
                        result = model.fit(frame)
                        elapsed = time.perf_counter() - start
                        predicted = self._extract_predicted_edges(result["graph"].edges())
                        metrics = precision_recall_f1(predicted, true_edges)
                        records.append(
                            {
                                "n_nodes": n_nodes,
                                "n_obs": n_obs,
                                "graph_id": graph_id,
                                "ci_method": method,
                                "precision": metrics["precision"],
                                "recall": metrics["recall"],
                                "f1": metrics["f1"],
                                "shd": shd(predicted, true_edges),
                                "runtime_sec": elapsed,
                            }
                        )
        self._save_outputs(pd.DataFrame(records))

    def _save_outputs(self, result_frame: pd.DataFrame) -> None:
        result_frame.to_csv(self.paths.results_tables_dir / "simulation_metrics.csv", index=False)
        summary = (
            result_frame.groupby(["ci_method", "n_nodes", "n_obs"], as_index=False)[["f1", "runtime_sec"]]
            .mean()
            .sort_values(["n_obs", "n_nodes"])
        )
        summary.to_csv(self.paths.results_tables_dir / "simulation_summary.csv", index=False)
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=summary, x="n_nodes", y="f1", hue="ci_method", style="n_obs", markers=True)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.paths.results_figures_dir / "simulation_f1.png", dpi=150)
        with open(self.paths.results_tables_dir / "simulation_config.json", "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "seed": self.config.seed,
                    "node_grid": self.config.node_grid,
                    "obs_grid": self.config.obs_grid,
                    "n_graphs_each": self.config.n_graphs,
                },
                file_obj,
                indent=2,
            )
        LOGGER.info("Saved simulation outputs under %s", self.paths.results_tables_dir)


def run(quick: bool = False):
    config = SimulationConfig(
        seed=0,
        node_grid=[3, 5] if quick else [3, 4, 5, 6, 8, 10, 15],
        obs_grid=[50, 150] if quick else [50, 150, 300, 500, 1000],
        ci_methods=["parcorr", "kcit_hbe", "rcot_hbe", "cmiknn"],
        n_graphs=5 if quick else 50,
    )
    SimulationExperiment(ProjectPaths(Path(".")), config).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a lightweight sanity run.")
    args = parser.parse_args()
    run(quick=args.quick)

