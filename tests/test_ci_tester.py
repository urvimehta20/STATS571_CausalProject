from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.cdnots.ci_tests import CITester


class CITesterTests(unittest.TestCase):
    def test_fallback_methods_are_labeled(self) -> None:
        data = pd.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, 30),
                "y": np.linspace(0.1, 1.1, 30),
                "z": np.linspace(-1.0, 1.0, 30),
            }
        )
        tester = CITester(method="kcit_hbe", alpha=0.05)
        result = tester.test(data, "x", "y", ["z"])
        self.assertEqual(result.method, "kcit_hbe_fallback_parcorr")
        self.assertIn("tests_run", tester.get_stability_summary())
        self.assertEqual(tester.get_stability_summary()["tests_run"], 1)

    def test_insufficient_rows_counter_increments(self) -> None:
        data = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [2.0, 3.0],
                "z": [1.0, 1.0],
            }
        )
        tester = CITester(method="parcorr", alpha=0.05)
        result = tester.test(data, "x", "y", ["z"])
        self.assertEqual(result.p_value, 1.0)
        self.assertEqual(tester.get_stability_summary()["insufficient_rows"], 1)

    def test_constant_series_is_handled_without_nan(self) -> None:
        data = pd.DataFrame(
            {
                "x": np.ones(20),
                "y": np.arange(20, dtype=float),
            }
        )
        tester = CITester(method="parcorr", alpha=0.05)
        result = tester.test(data, "x", "y")
        self.assertTrue(np.isfinite(result.p_value))
        self.assertTrue(np.isfinite(result.statistic))
        self.assertGreaterEqual(tester.get_stability_summary()["degenerate_pearson"], 1)


if __name__ == "__main__":
    unittest.main()
