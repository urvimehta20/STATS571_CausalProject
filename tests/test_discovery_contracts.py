from __future__ import annotations

import unittest

import numpy as np

from discovery2.graph_decode import decompose_adjacency, edges_from_adjacency


class DiscoveryContractTests(unittest.TestCase):
    def test_discovery_directed_edge_schema_contract(self) -> None:
        adjacency = np.array(
            [
                [0, -1, 0],
                [1, 0, -1],
                [0, 1, 0],
            ],
            dtype=float,
        )
        names = ["A", "B", "context_time"]
        edge_table = edges_from_adjacency(adjacency, names)
        directed, undirected = decompose_adjacency(adjacency, names)

        self.assertTrue({"from", "to", "weight"} <= set(edge_table.columns))
        self.assertIn(("A", "B"), directed)
        self.assertIn(("B", "context_time"), directed)
        self.assertEqual(len(undirected), 0)


if __name__ == "__main__":
    unittest.main()
