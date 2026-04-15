from __future__ import annotations

import unittest

import pandas as pd

from experiments.causal_diagnostics import candidate_controls_from_graph, graph_ambiguity_flag


class CausalDiagnosticsTests(unittest.TestCase):
    def test_candidate_controls_parents_mode(self) -> None:
        directed = pd.DataFrame(
            {
                "from": ["A", "C", "B"],
                "to": ["Z", "Z", "Y"],
            }
        )
        controls = candidate_controls_from_graph(
            treatment="Z",
            outcome="Y",
            directed_edges=directed,
            mode="parents",
            available_columns=["A", "B", "C", "Z", "Y"],
        )
        self.assertEqual(sorted(controls), ["A", "C"])

    def test_graph_ambiguity_flag_detects_overlap(self) -> None:
        undirected = pd.DataFrame({"node_a": ["X"], "node_b": ["Z"]})
        flag, reason = graph_ambiguity_flag(
            treatment="Z",
            outcome="Y",
            undirected_edges=undirected,
            selected_controls=["A"],
        )
        self.assertTrue(flag)
        self.assertIn("Undirected adjacency", reason)


if __name__ == "__main__":
    unittest.main()
