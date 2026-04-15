from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from discovery2.services import CDNODPipeline, DatasetPreprocessor, PreparedDataset


class DiscoveryServicesTests(unittest.TestCase):
    def test_prepare_famafrench_adds_context_node(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "famafrench_apple_daily.csv"
            pd.DataFrame(
                {
                    "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                    "Mkt_RF": [0.1, 0.2, 0.3],
                    "SMB": [0.0, 0.1, 0.2],
                    "HML": [0.2, 0.3, 0.4],
                    "RMW": [0.1, 0.1, 0.1],
                    "CMA": [0.0, 0.0, 0.0],
                    "RF": [0.01, 0.01, 0.01],
                    "AAPL_RET": [1.2, -0.5, 0.3],
                }
            ).to_csv(csv_path, index=False)
            prepared = DatasetPreprocessor.prepare_famafrench(csv_path, max_rows=None)
            self.assertEqual(prepared.run_name, "famafrench")
            self.assertEqual(prepared.data.shape[0], 3)
            self.assertEqual(prepared.context_index.shape, (3, 1))
            self.assertEqual(prepared.node_names[-1], "context_time")

    def test_prepare_macro_country_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "macro_countries_monthly.csv"
            pd.DataFrame(
                {
                    "date": ["2020-01-31", "2020-01-31", "2020-02-29"],
                    "country": ["US", "CA", "US"],
                    "unemployment": [4.0, 6.0, 3.9],
                    "cpi": [260.0, 140.0, 261.0],
                    "ppi": [200.0, 120.0, 200.5],
                }
            ).to_csv(csv_path, index=False)
            prepared = DatasetPreprocessor.prepare_macro(csv_path, max_rows=None, country="US")
            self.assertEqual(prepared.run_name, "macro_US")
            self.assertEqual(prepared.data.shape[0], 2)
            self.assertEqual(prepared.node_names[-1], "context_country")

    def test_pipeline_rejects_too_few_rows(self) -> None:
        prepared = PreparedDataset(
            data=np.random.randn(5, 3),
            context_index=np.arange(5, dtype=float).reshape(-1, 1),
            node_names=["A", "B", "C", "context_time"],
            run_name="tiny",
        )
        pipeline = CDNODPipeline(alpha=0.05)
        with self.assertRaises(ValueError):
            pipeline.run(prepared)


if __name__ == "__main__":
    unittest.main()
