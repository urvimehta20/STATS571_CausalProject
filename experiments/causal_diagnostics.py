from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class IdentificationDiagnostics:
    treatment: str
    outcome: str
    candidate_controls: list[str]
    graph_ambiguity_flag: bool
    ambiguity_reason: str
    n_rows_used: int


def build_parent_map(directed_edges: pd.DataFrame) -> dict[str, set[str]]:
    parent_map: dict[str, set[str]] = {}
    for _, row in directed_edges.iterrows():
        source = str(row["from"])
        target = str(row["to"])
        parent_map.setdefault(target, set()).add(source)
        parent_map.setdefault(source, set())
    return parent_map


def candidate_controls_from_graph(
    *,
    treatment: str,
    outcome: str,
    directed_edges: pd.DataFrame,
    mode: str,
    available_columns: Iterable[str],
) -> list[str]:
    columns = set(str(column) for column in available_columns)
    parent_map = build_parent_map(directed_edges)

    treatment_parents = sorted(parent_map.get(treatment, set()))
    if mode == "parents":
        candidates = treatment_parents
    elif mode == "minimal_backdoor":
        # Heuristic backdoor candidate set:
        # parents(Z) plus shared parents of treatment/outcome.
        outcome_parents = set(parent_map.get(outcome, set()))
        shared = sorted(set(treatment_parents).intersection(outcome_parents))
        candidates = sorted(set(treatment_parents).union(shared))
    else:
        raise ValueError(f"Unsupported adjustment mode: {mode}")

    skip = {treatment, outcome, "Date", "date", "country"}
    return [name for name in candidates if name in columns and name not in skip]


def graph_ambiguity_flag(
    *,
    treatment: str,
    outcome: str,
    undirected_edges: pd.DataFrame | None,
    selected_controls: list[str],
) -> tuple[bool, str]:
    if undirected_edges is None or undirected_edges.empty:
        return False, ""
    watch_nodes = {treatment, outcome, *selected_controls}
    for _, row in undirected_edges.iterrows():
        node_a = str(row.get("node_a", ""))
        node_b = str(row.get("node_b", ""))
        if node_a in watch_nodes or node_b in watch_nodes:
            return True, f"Undirected adjacency near identification set: ({node_a}, {node_b})"
    return False, ""

