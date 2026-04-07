from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.cdnots.core import CDNOTS, CDNOTSConfig


PERIODS = [
    ("2000-01-01", "2007-12-31", "2000_2007"),
    ("2008-01-01", "2015-12-31", "2008_2015"),
    ("2016-01-01", "2022-12-31", "2016_2022"),
]


def run():
    in_path = Path("data/raw/famafrench_apple_daily.csv")
    if not in_path.exists():
        raise FileNotFoundError("Run scripts/download_famafrench_apple.py first.")
    df = pd.read_csv(in_path, index_col=0, parse_dates=True)
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = Path("results/figures")
    figs.mkdir(parents=True, exist_ok=True)
    edge_rows = []
    for start, end, tag in PERIODS + [("2000-01-01", "2022-12-31", "full")]:
        cut = df.loc[start:end].dropna().copy()
        if cut.empty:
            continue
        cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "AAPL_RET"]
        model = CDNOTS(CDNOTSConfig(max_lag=4, ci_method="kcit_hbe", alpha=0.05, max_condition_set=2))
        res = model.fit(cut[cols])
        for u, v in res["graph"].edges():
            edge_rows.append({"period": tag, "from": u, "to": v})
    pd.DataFrame(edge_rows).to_csv(out_dir / "case_famafrench_apple_edges.csv", index=False)

    df[["Mkt_RF", "SMB", "HML", "RMW", "CMA", "AAPL_RET"]].plot(subplots=True, figsize=(10, 9), legend=False)
    plt.tight_layout()
    plt.savefig(figs / "case_famafrench_apple_series.png", dpi=150)


if __name__ == "__main__":
    run()

