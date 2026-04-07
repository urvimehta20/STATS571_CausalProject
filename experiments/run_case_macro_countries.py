from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cdnots.core import CDNOTS, CDNOTSConfig


def preprocess_country(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    out["cpi_change"] = out["cpi"].pct_change() * 100.0
    out["ppi_change"] = out["ppi"].pct_change() * 100.0
    out = out[["unemployment", "cpi_change", "ppi_change"]].dropna()
    return out


def run():
    in_path = Path("data/raw/macro_countries_monthly.csv")
    if not in_path.exists():
        raise FileNotFoundError("Run scripts/download_macro_data.py first.")
    df = pd.read_csv(in_path, parse_dates=["date"])
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for country, sub in df.groupby("country"):
        pre = preprocess_country(sub)
        if len(pre) < 60:
            continue
        model = CDNOTS(CDNOTSConfig(max_lag=1, ci_method="parcorr", alpha=0.05))
        g = model.fit(pre)["graph"]
        for u, v in g.edges():
            rows.append({"scope": country, "from": u, "to": v})

    pooled = df.copy()
    pooled["country_code"] = pooled["country"].astype("category").cat.codes
    pooled["cpi_change"] = pooled["cpi"].pct_change() * 100.0
    pooled["ppi_change"] = pooled["ppi"].pct_change() * 100.0
    pooled = pooled[["unemployment", "cpi_change", "ppi_change", "country_code"]].dropna()
    g = CDNOTS(CDNOTSConfig(max_lag=1, ci_method="kcit_hbe", alpha=0.05)).fit(pooled)["graph"]
    for u, v in g.edges():
        rows.append({"scope": "pooled", "from": u, "to": v})

    pd.DataFrame(rows).to_csv(out_dir / "case_macro_edges.csv", index=False)


if __name__ == "__main__":
    run()

