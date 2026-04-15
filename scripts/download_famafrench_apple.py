from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from src.cdnots.project_io import ProjectPaths, get_logger

LOGGER = get_logger("scripts.download_famafrench_apple")


def _download_famafrench_5f_daily() -> pd.DataFrame:
    """
    Download and parse the Fama-French 5-factor daily dataset directly from
    Kenneth French's public ZIP endpoint.
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        inner_name = archive.namelist()[0]
        raw_text = archive.read(inner_name).decode("utf-8", errors="ignore")

    lines = raw_text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(",Mkt-RF"):
            start_idx = idx
            break
    if start_idx is None:
        raise RuntimeError("Could not locate Fama-French daily table header.")

    table_lines: list[str] = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            break
        # Daily rows have YYYYMMDD in first column
        if stripped[0].isdigit() and len(stripped.split(",")[0]) == 8:
            table_lines.append(stripped)
        elif stripped.startswith(",Mkt-RF"):
            table_lines.append(stripped)
        elif table_lines:
            break

    ff = pd.read_csv(io.StringIO("\n".join(table_lines)))
    ff = ff.rename(columns={ff.columns[0]: "Date", "Mkt-RF": "Mkt_RF"})
    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
    ff = ff.set_index("Date").sort_index()
    return ff


def main() -> None:
    paths = ProjectPaths(Path("."))
    paths.ensure_standard_dirs()

    ff = _download_famafrench_5f_daily()

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

