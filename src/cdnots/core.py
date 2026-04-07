from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd

from .ci_tests import CITester
from .orientation import orient_lag_edges, orient_time_arrow, orient_v_structures
from .utils import build_lagged_frame, powerset_limited


@dataclass
class CDNOTSConfig:
    max_lag: int = 1
    alpha: float = 0.05
    ci_method: str = "parcorr"
    max_condition_set: int = 2


class CDNOTS:
    def __init__(self, config: Optional[CDNOTSConfig] = None):
        self.config = config or CDNOTSConfig()
        self.ci = CITester(method=self.config.ci_method, alpha=self.config.alpha)

    def fit(self, df: pd.DataFrame) -> Dict[str, object]:
        lagged = build_lagged_frame(df, self.config.max_lag)
        graph, sepsets = self._skeleton_discovery(lagged)
        graph = self._stage3_orientation(graph, sepsets)
        graph = self._stage4_orientation(graph, lagged)
        return {
            "graph": graph,
            "sepsets": sepsets,
            "lagged_data": lagged,
        }

    def _skeleton_discovery(self, lagged: pd.DataFrame):
        nodes = list(lagged.columns)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        for i, x in enumerate(nodes):
            for y in nodes[i + 1 :]:
                g.add_edge(x, y)

        sepsets: Dict[Tuple[str, str], Set[str]] = {}
        changed = True
        while changed:
            changed = False
            for x, y in list(g.edges()):
                neighbors = set(g.neighbors(x)).union(g.neighbors(y)) - {x, y}
                removed = False
                for zset in powerset_limited(sorted(neighbors), self.config.max_condition_set):
                    test = self.ci.test(lagged, x, y, list(zset))
                    if test.p_value > self.config.alpha:
                        g.remove_edge(x, y)
                        sepsets[tuple(sorted((x, y)))] = set(zset)
                        changed = True
                        removed = True
                        break
                if removed:
                    continue

        dg = nx.DiGraph()
        dg.add_nodes_from(g.nodes())
        for u, v in g.edges():
            dg.add_edge(u, v)
            dg.add_edge(v, u)
        return dg, sepsets

    def _stage3_orientation(self, g: nx.DiGraph, sepsets: Dict[Tuple[str, str], Set[str]]) -> nx.DiGraph:
        lag_map = {}
        for n in g.nodes():
            if n == "T":
                continue
            lag_map[n] = int(n.split("_t-")[1]) if "_t-" in n else 0
        out = orient_time_arrow(g)
        out = orient_lag_edges(out, lag_map)
        out = orient_v_structures(out, sepsets)
        return out

    def _stage4_orientation(self, g: nx.DiGraph, _lagged: pd.DataFrame) -> nx.DiGraph:
        # Placeholder for module-change based orientation. We apply a Meek-like closure:
        # if X->Y and Y-Z undirected and X not adjacent Z, orient Y->Z.
        out = g.copy()
        made_change = True
        while made_change:
            made_change = False
            for y, z in list(out.edges()):
                if not out.has_edge(z, y):
                    continue
                for x in list(out.predecessors(y)):
                    if out.has_edge(y, x):
                        continue
                    if out.has_edge(x, z) or out.has_edge(z, x):
                        continue
                    if out.has_edge(z, y):
                        out.remove_edge(z, y)
                        made_change = True
        return out

