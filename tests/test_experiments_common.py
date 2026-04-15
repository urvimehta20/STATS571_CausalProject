from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.common import build_paths, edges_to_rows, load_famafrench_daily, load_macro_monthly


class ExperimentsCommonTests(unittest.TestCase):
    def test_edges_to_rows_builds_expected_schema(self) -> None:
        rows = edges_to_rows([("X", "Y"), ("Y", "Z")], key_name="scope", key_value="test")
        self.assertEqual(rows[0]["scope"], "test")
        self.assertEqual(rows[0]["from"], "X")
        self.assertEqual(rows[0]["to"], "Y")
        self.assertEqual(len(rows), 2)

    def test_loaders_read_written_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = build_paths(root)

            ff_path = paths.raw_data_dir / "famafrench_apple_daily.csv"
            macro_path = paths.raw_data_dir / "macro_countries_monthly.csv"

            pd.DataFrame(
                {
                    "Date": ["2020-01-01", "2020-01-02"],
                    "Mkt_RF": [0.1, 0.2],
                    "SMB": [0.0, 0.1],
                    "HML": [0.2, 0.3],
                    "RMW": [0.1, 0.1],
                    "CMA": [0.0, 0.0],
                    "AAPL_RET": [1.2, -0.5],
                }
            ).to_csv(ff_path, index=False)

            pd.DataFrame(
                {
                    "date": ["2020-01-31", "2020-02-29"],
                    "country": ["US", "US"],
                    "unemployment": [4.0, 3.9],
                    "cpi": [260.0, 261.0],
                    "ppi": [200.0, 200.5],
                }
            ).to_csv(macro_path, index=False)

            ff_loaded = load_famafrench_daily(paths)
            macro_loaded = load_macro_monthly(paths)

            self.assertEqual(len(ff_loaded), 2)
            self.assertEqual(len(macro_loaded), 2)


if __name__ == "__main__":
    unittest.main()
