from __future__ import annotations

import unittest

import pandas as pd

from experiments.run_graph_falsification import _build_parent_map


class GraphFalsificationTests(unittest.TestCase):
    def test_build_parent_map_collects_parents_and_nodes(self) -> None:
        directed = pd.DataFrame(
            {
                "from": ["A", "B", "A"],
                "to": ["C", "C", "D"],
            }
        )
        parent_map = _build_parent_map(directed)
        self.assertEqual(parent_map["C"], {"A", "B"})
        self.assertEqual(parent_map["D"], {"A"})
        self.assertIn("A", parent_map)


if __name__ == "__main__":
    unittest.main()
