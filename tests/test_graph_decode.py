from __future__ import annotations

import unittest

import numpy as np

from discovery2.graph_decode import decompose_adjacency, edges_from_adjacency


class GraphDecodeTests(unittest.TestCase):
    def test_edges_from_adjacency_emits_weighted_rows(self) -> None:
        adjacency = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        names = ["A", "B", "context_time"]
        edges = edges_from_adjacency(adjacency, names)
        self.assertEqual(set(edges.columns), {"from", "to", "weight"})
        self.assertGreaterEqual(len(edges), 4)

    def test_decompose_adjacency_directed_and_undirected(self) -> None:
        adjacency = np.array(
            [
                [0.0, -1.0, -1.0],
                [1.0, 0.0, -1.0],
                [-1.0, -1.0, 0.0],
            ]
        )
        names = ["A", "B", "C"]
        directed, undirected = decompose_adjacency(adjacency, names)
        self.assertIn(("A", "B"), directed)
        self.assertIn(("A", "C"), undirected)
        self.assertIn(("B", "C"), undirected)


if __name__ == "__main__":
    unittest.main()
