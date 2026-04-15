from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

import networkx as nx
import pandas as pd

from .ci_tests import CITester
from .orientation import orient_lag_edges, orient_time_arrow, orient_v_structures
from .utils import powerset_limited


@dataclass
class SkeletonDiscoveryStage:
    """Build a partially directed skeleton using CI-based edge pruning."""

    ci_tester: CITester
    alpha: float
    max_condition_set: int

    def run(self, lagged: pd.DataFrame) -> tuple[nx.DiGraph, Dict[Tuple[str, str], Set[str]]]:
        nodes = list(lagged.columns)
        skeleton = nx.Graph()
        skeleton.add_nodes_from(nodes)
        for left_index, source in enumerate(nodes):
            for target in nodes[left_index + 1 :]:
                skeleton.add_edge(source, target)

        sepsets: Dict[Tuple[str, str], Set[str]] = {}
        changed = True
        while changed:
            changed = False
            for source, target in list(skeleton.edges()):
                neighbors = set(skeleton.neighbors(source)).union(skeleton.neighbors(target))
                neighbors -= {source, target}
                for condition_set in powerset_limited(sorted(neighbors), self.max_condition_set):
                    test_result = self.ci_tester.test(lagged, source, target, list(condition_set))
                    if test_result.p_value > self.alpha:
                        skeleton.remove_edge(source, target)
                        sepsets[tuple(sorted((source, target)))] = set(condition_set)
                        changed = True
                        break

        output = nx.DiGraph()
        output.add_nodes_from(skeleton.nodes())
        for source, target in skeleton.edges():
            output.add_edge(source, target)
            output.add_edge(target, source)
        return output, sepsets


class OrientationStage:
    """Apply stage-3 and stage-4 orientation rules."""

    @staticmethod
    def run_stage3(
        graph: nx.DiGraph,
        sepsets: Dict[Tuple[str, str], Set[str]],
    ) -> nx.DiGraph:
        lag_map: Dict[str, int] = {}
        for node in graph.nodes():
            if node == "T":
                continue
            lag_map[node] = int(node.split("_t-")[1]) if "_t-" in node else 0

        oriented = orient_time_arrow(graph)
        oriented = orient_lag_edges(oriented, lag_map)
        oriented = orient_v_structures(oriented, sepsets)
        return oriented

    @staticmethod
    def run_stage4(graph: nx.DiGraph) -> nx.DiGraph:
        # Meek-like closure: X->Y and Y-Z undirected and X not adjacent Z => Y->Z.
        output = graph.copy()
        made_change = True
        while made_change:
            made_change = False
            for node_y, node_z in list(output.edges()):
                if not output.has_edge(node_z, node_y):
                    continue
                for node_x in list(output.predecessors(node_y)):
                    if output.has_edge(node_y, node_x):
                        continue
                    if output.has_edge(node_x, node_z) or output.has_edge(node_z, node_x):
                        continue
                    if output.has_edge(node_z, node_y):
                        output.remove_edge(node_z, node_y)
                        made_change = True
        return output
