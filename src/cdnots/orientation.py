from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

import networkx as nx


def orient_time_arrow(g: nx.DiGraph) -> nx.DiGraph:
    out = g.copy()
    for u, v in list(out.edges()):
        if u == "T" and v != "T":
            continue
        if v == "T" and u != "T":
            out.remove_edge(v, u)
            out.add_edge("T", u)
    return out


def orient_lag_edges(g: nx.DiGraph, lag_map: Dict[str, int]) -> nx.DiGraph:
    out = g.copy()
    for u, v in list(out.edges()):
        if u not in lag_map or v not in lag_map:
            continue
        if lag_map[u] > lag_map[v]:
            if out.has_edge(v, u):
                out.remove_edge(v, u)
        elif lag_map[v] > lag_map[u]:
            if out.has_edge(u, v):
                out.remove_edge(u, v)
    return out


def orient_v_structures(g: nx.DiGraph, sepsets: Dict[Tuple[str, str], Set[str]]) -> nx.DiGraph:
    out = g.copy()
    nodes = list(out.nodes())
    for z in nodes:
        pa = list(out.predecessors(z))
        for i in range(len(pa)):
            for j in range(i + 1, len(pa)):
                x, y = pa[i], pa[j]
                if out.has_edge(x, y) or out.has_edge(y, x):
                    continue
                sep = sepsets.get(tuple(sorted((x, y))), set())
                if z not in sep:
                    if out.has_edge(z, x):
                        out.remove_edge(z, x)
                    if out.has_edge(z, y):
                        out.remove_edge(z, y)
                    out.add_edge(x, z)
                    out.add_edge(y, z)
    return out

