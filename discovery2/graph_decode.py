from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def edges_from_adjacency(adjacency: np.ndarray, node_names: list[str]) -> pd.DataFrame:
    """Convert adjacency matrix into a long edge table."""
    rows: list[dict[str, float | str]] = []
    num_nodes = adjacency.shape[0]
    for source_idx in range(num_nodes):
        for target_idx in range(num_nodes):
            if source_idx == target_idx:
                continue
            if adjacency[source_idx, target_idx] != 0:
                rows.append(
                    {
                        "from": node_names[source_idx],
                        "to": node_names[target_idx],
                        "weight": float(adjacency[source_idx, target_idx]),
                    }
                )
    return pd.DataFrame(rows)


def decompose_adjacency(adjacency: np.ndarray, node_names: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Decode causal-learn endpoints into directed and undirected edge sets."""
    num_nodes = adjacency.shape[0]
    directed: list[tuple[str, str]] = []
    undirected: list[tuple[str, str]] = []
    for source_idx in range(num_nodes):
        for target_idx in range(source_idx + 1, num_nodes):
            source_to_target = float(adjacency[source_idx, target_idx])
            target_to_source = float(adjacency[target_idx, source_idx])
            if source_to_target == 0 and target_to_source == 0:
                continue
            source = node_names[source_idx]
            target = node_names[target_idx]
            if source_to_target == -1 and target_to_source == 1:
                directed.append((source, target))
            elif source_to_target == 1 and target_to_source == -1:
                directed.append((target, source))
            else:
                undirected.append((source, target))
    return directed, undirected


def save_fallback_png(adjacency: np.ndarray, node_names: list[str], out_png: str) -> None:
    """Render graph with NetworkX when Graphviz is not available."""
    directed, undirected = decompose_adjacency(adjacency, node_names)
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(node_names)
    layout_graph.add_edges_from(undirected)
    layout_graph.add_edges_from(directed)
    position = nx.spring_layout(layout_graph, seed=42, k=2.0 / max(np.sqrt(len(node_names)), 1))

    plt.figure(figsize=(12, 8))
    axis = plt.gca()
    nx.draw_networkx_nodes(
        layout_graph,
        position,
        nodelist=node_names,
        node_size=1400,
        node_color="#E8F1FF",
        edgecolors="#335C99",
        ax=axis,
    )
    nx.draw_networkx_labels(layout_graph, position, labels={name: name for name in node_names}, font_size=9, ax=axis)
    if undirected:
        nx.draw_networkx_edges(nx.Graph(undirected), position, width=2.2, edge_color="#6B6B6B", ax=axis)
    if directed:
        nx.draw_networkx_edges(
            nx.DiGraph(directed),
            position,
            width=2.0,
            edge_color="#1A1A1A",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            connectionstyle="arc3,rad=0.08",
            ax=axis,
        )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
