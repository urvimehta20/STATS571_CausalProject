from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.cdnots.core import CDNOTS, CDNOTSConfig
from src.cdnots.project_io import get_logger
from experiments.common import build_paths, edges_to_rows, load_famafrench_daily

LOGGER = get_logger("experiments.run_case_famafrench_apple")


PERIODS = [
    ("2000-01-01", "2007-12-31", "2000_2007"),
    ("2008-01-01", "2015-12-31", "2008_2015"),
    ("2016-01-01", "2022-12-31", "2016_2022"),
]


def run():
    paths = build_paths(Path("."))
    df = load_famafrench_daily(paths, index_by_date=True)
    edge_rows = []
    selected_columns = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "AAPL_RET"]
    for start, end, tag in PERIODS + [("2000-01-01", "2022-12-31", "full")]:
        cut = df.loc[start:end].dropna().copy()
        if cut.empty:
            continue
        model = CDNOTS(CDNOTSConfig(max_lag=4, ci_method="kcit_hbe", alpha=0.05, max_condition_set=2))
        result = model.fit(cut[selected_columns])
        edge_rows.extend(edges_to_rows(result["graph"].edges(), key_name="period", key_value=tag))
    edges_path = paths.results_tables_dir / "case_famafrench_apple_edges.csv"
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)

    df[selected_columns].plot(subplots=True, figsize=(10, 9), legend=False)
    plt.tight_layout()
    figure_path = paths.results_figures_dir / "case_famafrench_apple_series.png"
    plt.savefig(figure_path, dpi=150)
    LOGGER.info("Saved case-study outputs to %s and %s", edges_path, figure_path)


if __name__ == "__main__":
    run()

