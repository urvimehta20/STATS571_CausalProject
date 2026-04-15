from __future__ import annotations

import unittest

from experiments.lecture13_graph_adjustment import _hidden_confounding_sensitivity


class Lecture13AdjustmentTests(unittest.TestCase):
    def test_hidden_confounding_sensitivity_increases_with_t_stat(self) -> None:
        low_partial, low_rv = _hidden_confounding_sensitivity(t_stat=1.0, dof=100.0)
        high_partial, high_rv = _hidden_confounding_sensitivity(t_stat=5.0, dof=100.0)
        self.assertGreater(high_partial, low_partial)
        self.assertGreater(high_rv, low_rv)

    def test_hidden_confounding_sensitivity_handles_nonpositive_dof(self) -> None:
        partial_r2, robustness = _hidden_confounding_sensitivity(t_stat=2.0, dof=0.0)
        self.assertEqual(partial_r2, 0.0)
        self.assertEqual(robustness, 0.0)


if __name__ == "__main__":
    unittest.main()
