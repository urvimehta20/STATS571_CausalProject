from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas_datareader import data as pdr
from src.cdnots.project_io import ProjectPaths, get_logger

LOGGER = get_logger("scripts.download_macro_data")


FRED_SERIES = {
    "US": {"UNRATE": "unemployment", "CPIAUCSL": "cpi", "PPIACO": "ppi"},
    "CA": {"LRUN64TTCAM156S": "unemployment", "CPALCY01CAM661N": "cpi", "PPIQ": "ppi"},
    "JP": {"LRUN64TTJPM156S": "unemployment", "JPNCPIALLMINMEI": "cpi", "JPNPPIENGMISM": "ppi"},
    "FR": {"LRUN64TTFRM156S": "unemployment", "FRACPIALLMINMEI": "cpi", "FRAPPIENGMISM": "ppi"},
    "GB": {"LRUN64TTGBM156S": "unemployment", "GBRCPIALLMINMEI": "cpi", "GBRPPIENGMISM": "ppi"},
    "IT": {"LRUN64TTITM156S": "unemployment", "ITACPIALLMINMEI": "cpi", "ITAPPIENGMISM": "ppi"},
}


def main() -> None:
    paths = ProjectPaths(Path("."))
    paths.ensure_standard_dirs()
    start, end = "2000-01-01", "2024-12-01"
    rows = []
    for country, mapping in FRED_SERIES.items():
        country_df = pd.DataFrame()
        for fred_code, col in mapping.items():
            try:
                s = pdr.DataReader(fred_code, "fred", start, end).rename(columns={fred_code: col})
                country_df = s if country_df.empty else country_df.join(s, how="outer")
            except Exception:
                continue
        if country_df.empty:
            continue
        country_df = country_df.sort_index().resample("ME").last()
        country_df["country"] = country
        rows.append(country_df.reset_index().rename(columns={"DATE": "date", "index": "date"}))
    if not rows:
        raise RuntimeError("No macroeconomic series could be downloaded from FRED.")
    final = pd.concat(rows, ignore_index=True)
    output_path = paths.raw_data_dir / "macro_countries_monthly.csv"
    final.to_csv(output_path, index=False)
    LOGGER.info("Saved %s rows to %s", len(final), output_path)


if __name__ == "__main__":
    main()

