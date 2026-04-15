from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.run_macro_policy_case import _synthetic_weights


class MacroPolicyCaseTests(unittest.TestCase):
    def test_synthetic_weights_nonnegative_and_sum_to_one(self) -> None:
        treated = pd.Series([1.0, 1.2, 1.1])
        donors = pd.DataFrame(
            {
                "C1": [1.0, 1.0, 1.0],
                "C2": [0.5, 0.8, 0.7],
            }
        )
        weights = _synthetic_weights(treated, donors)
        self.assertTrue(np.all(weights.values >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
