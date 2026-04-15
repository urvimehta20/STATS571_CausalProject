from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from src.cdnots.ci_tests import CITester
from src.cdnots.stages import OrientationStage, SkeletonDiscoveryStage


class CDNOTSStagesTests(unittest.TestCase):
    def test_skeleton_stage_outputs_bidirectional_graph(self) -> None:
        lagged = pd.DataFrame(
            {
                "X0_t": [0.0, 0.1, 0.2, 0.3, 0.4],
                "X1_t": [0.0, -0.1, 0.1, -0.2, 0.2],
                "T": [0, 1, 2, 3, 4],
            }
        )
        stage = SkeletonDiscoveryStage(
            ci_tester=CITester(method="parcorr", alpha=0.05),
            alpha=0.05,
            max_condition_set=1,
        )
        graph, sepsets = stage.run(lagged)
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertIsInstance(sepsets, dict)
        for source, target in graph.edges():
            self.assertTrue(graph.has_edge(target, source) or source == target)

    def test_orientation_stage3_respects_time_direction(self) -> None:
        graph = nx.DiGraph()
        graph.add_nodes_from(["X_t", "X_t-1", "T"])
        graph.add_edge("X_t", "T")
        graph.add_edge("T", "X_t")
        graph.add_edge("X_t", "X_t-1")
        graph.add_edge("X_t-1", "X_t")
        oriented = OrientationStage.run_stage3(graph, sepsets={})
        self.assertTrue(oriented.has_edge("T", "X_t"))
        self.assertFalse(oriented.has_edge("X_t", "T"))
        self.assertTrue(oriented.has_edge("X_t-1", "X_t"))
        self.assertFalse(oriented.has_edge("X_t", "X_t-1"))


if __name__ == "__main__":
    unittest.main()
