from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.GraphUtils import GraphUtils

from .graph_decode import decompose_adjacency, edges_from_adjacency


@dataclass
class PreparedDataset:
    """Container for preprocessed data passed to CD-NOD."""

    data: np.ndarray
    context_index: np.ndarray
    node_names: list[str]
    run_name: str


@dataclass
class DiscoveryArtifacts:
    """Container for discovery outputs before persistence."""

    adjacency: np.ndarray
    edges: pd.DataFrame
    directed_edges: list[tuple[str, str]]
    undirected_edges: list[tuple[str, str]]
    node_names: list[str]
    dot_graph: str


class DatasetPreprocessor:
    """Factory class for dataset-specific preprocessing."""

    @staticmethod
    def prepare_famafrench(raw_path: Path, max_rows: int | None) -> PreparedDataset:
        frame = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        if max_rows:
            frame = frame.tail(max_rows).reset_index(drop=True)

        feature_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF", "AAPL_RET"]
        feature_cols = [column for column in feature_cols if column in frame.columns]
        selected = frame[feature_cols].dropna().reset_index(drop=True)
        context = np.arange(len(selected), dtype=float).reshape(-1, 1)
        return PreparedDataset(
            data=selected.to_numpy(dtype=float),
            context_index=context,
            node_names=selected.columns.tolist() + ["context_time"],
            run_name="famafrench",
        )

    @staticmethod
    def prepare_macro(raw_path: Path, max_rows: int | None, country: str) -> PreparedDataset:
        frame = pd.read_csv(raw_path, parse_dates=["date"]).sort_values(["country", "date"]).reset_index(drop=True)
        if country.lower() != "all":
            frame = frame[frame["country"] == country].copy()
        if max_rows:
            frame = frame.tail(max_rows).reset_index(drop=True)

        numeric = frame[["unemployment", "cpi", "ppi"]].astype(float)
        valid_rows = numeric.notna().all(axis=1)
        numeric = numeric.loc[valid_rows].reset_index(drop=True)
        countries = frame.loc[valid_rows, "country"].reset_index(drop=True)
        country_codes = pd.Categorical(countries).codes.astype(float).reshape(-1, 1)
        return PreparedDataset(
            data=numeric.to_numpy(dtype=float),
            context_index=country_codes,
            node_names=numeric.columns.tolist() + ["context_country"],
            run_name=f"macro_{country}",
        )


class CDNODPipeline:
    """Application service for running CD-NOD end-to-end."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def run(self, prepared: PreparedDataset) -> DiscoveryArtifacts:
        if prepared.data.shape[0] < 10:
            raise ValueError(f"Not enough rows after preprocessing: {prepared.data.shape[0]}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            learned_graph = cdnod(prepared.data, prepared.context_index, alpha=self.alpha)

        adjacency = np.asarray(learned_graph.G.graph)
        directed, undirected = decompose_adjacency(adjacency, prepared.node_names)
        return DiscoveryArtifacts(
            adjacency=adjacency,
            edges=edges_from_adjacency(adjacency, prepared.node_names),
            directed_edges=directed,
            undirected_edges=undirected,
            node_names=prepared.node_names,
            dot_graph=GraphUtils.to_pydot(learned_graph.G, labels=prepared.node_names).to_string(),
        )
