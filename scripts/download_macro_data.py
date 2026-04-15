from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
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


def _fetch_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch a single FRED series using the public CSV endpoint.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    series_df = pd.read_csv(io.StringIO(response.text))
    date_col = series_df.columns[0]
    series_df = series_df.rename(columns={date_col: "date", series_id: series_id})
    series_df["date"] = pd.to_datetime(series_df["date"], errors="coerce")
    series_df[series_id] = pd.to_numeric(series_df[series_id], errors="coerce")
    mask = (series_df["date"] >= pd.to_datetime(start)) & (series_df["date"] <= pd.to_datetime(end))
    return series_df.loc[mask].reset_index(drop=True)


def main() -> None:
    paths = ProjectPaths(Path("."))
    paths.ensure_standard_dirs()
    start, end = "2000-01-01", "2024-12-01"
    rows = []
    for country, mapping in FRED_SERIES.items():
        country_df = pd.DataFrame()
        for fred_code, col in mapping.items():
            try:
                s = _fetch_fred_series(fred_code, start, end).rename(columns={fred_code: col})
                s = s.set_index("date")
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

