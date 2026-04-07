from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench")[0]
    ff.index = pd.to_datetime(ff.index)
    ff = ff.rename(columns={"Mkt-RF": "Mkt_RF"})

    aapl = yf.download("AAPL", start="2000-01-01", end="2023-01-01", auto_adjust=True, progress=False)
    if isinstance(aapl.columns, pd.MultiIndex):
        aapl.columns = aapl.columns.get_level_values(0)
    aapl["AAPL_RET"] = aapl["Close"].pct_change() * 100.0
    aapl = aapl[["AAPL_RET"]].dropna()
    aapl.index = pd.to_datetime(aapl.index)

    merged = ff.join(aapl, how="inner").dropna()
    merged.to_csv(raw_dir / "famafrench_apple_daily.csv")
    print(f"Saved {len(merged)} rows to data/raw/famafrench_apple_daily.csv")


if __name__ == "__main__":
    main()

