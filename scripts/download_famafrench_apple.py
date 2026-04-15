from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from src.cdnots.project_io import ProjectPaths, get_logger

LOGGER = get_logger("scripts.download_famafrench_apple")


def main() -> None:
    paths = ProjectPaths(Path("."))
    paths.ensure_standard_dirs()

    ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench")[0]
    ff.index = pd.to_datetime(ff.index)
    ff = ff.rename(columns={"Mkt-RF": "Mkt_RF"})

    aapl = yf.download("AAPL", start="2000-01-01", end="2023-12-01", auto_adjust=True, progress=False)
    if isinstance(aapl.columns, pd.MultiIndex):
        aapl.columns = aapl.columns.get_level_values(0)
    aapl["AAPL_RET"] = aapl["Close"].pct_change() * 100.0
    aapl = aapl[["AAPL_RET"]].dropna()
    aapl.index = pd.to_datetime(aapl.index)

    merged = ff.join(aapl, how="inner").dropna()
    output_path = paths.raw_data_dir / "famafrench_apple_daily.csv"
    merged.to_csv(output_path)
    LOGGER.info("Saved %s rows to %s", len(merged), output_path)


if __name__ == "__main__":
    main()

