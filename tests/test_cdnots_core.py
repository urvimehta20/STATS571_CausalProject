from __future__ import annotations

import unittest

import pandas as pd

from src.cdnots.core import CDNOTS
from src.cdnots.models import CDNOTSConfig


class CDNOTSCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "X0": [0.0, 1.0, 0.2, -0.1, 0.4, -0.5, 0.8, -0.2, 0.1, 0.3, -0.4, 0.5],
                "X1": [0.5, 0.2, -0.3, 0.7, -0.1, 0.1, 0.2, 0.4, -0.6, 0.8, 0.5, -0.2],
            }
        )

    def test_fit_result_and_legacy_dict_contract(self) -> None:
        model = CDNOTS(CDNOTSConfig(max_lag=1, alpha=0.05, ci_method="parcorr", max_condition_set=1))
        typed = model.fit_result(self.frame)
        legacy = model.fit(self.frame)

        self.assertTrue(hasattr(typed, "graph"))
        self.assertIn("graph", legacy)
        self.assertIn("sepsets", legacy)
        self.assertIn("lagged_data", legacy)
        self.assertIn("ci_stability_summary", legacy)
        self.assertEqual(typed.graph.number_of_nodes(), legacy["graph"].number_of_nodes())

    def test_lagged_data_contains_time_node(self) -> None:
        model = CDNOTS(CDNOTSConfig(max_lag=1, alpha=0.05, ci_method="parcorr", max_condition_set=1))
        result = model.fit_result(self.frame)
        self.assertIn("T", result.lagged_data.columns)
        self.assertGreaterEqual(len(result.ci_stability_summary), 1)


if __name__ == "__main__":
    unittest.main()
