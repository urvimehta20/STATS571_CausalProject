from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.cdnots.project_io import ProjectPaths


def build_paths(root: Path | None = None) -> ProjectPaths:
    """Return project paths and ensure standard directories exist."""
    paths = ProjectPaths(root or Path("."))
    paths.ensure_standard_dirs()
    return paths


def require_file(path: Path, hint: str) -> None:
    """Raise a clear error when a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(hint)


def load_famafrench_daily(paths: ProjectPaths, index_by_date: bool = False) -> pd.DataFrame:
    """Load the Fama-French + Apple raw dataset."""
    in_path = paths.raw_data_dir / "famafrench_apple_daily.csv"
    require_file(in_path, "Run scripts/download_famafrench_apple.py first.")
    if index_by_date:
        return pd.read_csv(in_path, index_col=0, parse_dates=True)
    return pd.read_csv(in_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def load_macro_monthly(paths: ProjectPaths) -> pd.DataFrame:
    """Load the macro countries monthly dataset."""
    in_path = paths.raw_data_dir / "macro_countries_monthly.csv"
    require_file(in_path, "Run scripts/download_macro_data.py first.")
    return pd.read_csv(in_path, parse_dates=["date"])


def edges_to_rows(
    edges: Iterable[tuple[str, str]],
    *,
    key_name: str,
    key_value: str,
) -> list[dict[str, str]]:
    """Convert graph edges into a flat row list for CSV export."""
    return [{key_name: key_value, "from": source, "to": target} for source, target in edges]
