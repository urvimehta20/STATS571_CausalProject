from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cdnots.core import CDNOTS, CDNOTSConfig
from src.cdnots.project_io import get_logger
from experiments.common import build_paths, edges_to_rows, load_macro_monthly

LOGGER = get_logger("experiments.run_case_macro_countries")


def preprocess_country(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    out["cpi_change"] = out["cpi"].pct_change() * 100.0
    out["ppi_change"] = out["ppi"].pct_change() * 100.0
    out = out[["unemployment", "cpi_change", "ppi_change"]].dropna()
    return out


def run():
    paths = build_paths(Path("."))
    df = load_macro_monthly(paths)
    rows = []
    for country, sub in df.groupby("country"):
        pre = preprocess_country(sub)
        if len(pre) < 60:
            continue
        model = CDNOTS(CDNOTSConfig(max_lag=1, ci_method="parcorr", alpha=0.05))
        g = model.fit(pre)["graph"]
        rows.extend(edges_to_rows(g.edges(), key_name="scope", key_value=str(country)))

    pooled = df.copy()
    pooled["country_code"] = pooled["country"].astype("category").cat.codes
    pooled["cpi_change"] = pooled["cpi"].pct_change() * 100.0
    pooled["ppi_change"] = pooled["ppi"].pct_change() * 100.0
    pooled = pooled[["unemployment", "cpi_change", "ppi_change", "country_code"]].dropna()
    g = CDNOTS(CDNOTSConfig(max_lag=1, ci_method="kcit_hbe", alpha=0.05)).fit(pooled)["graph"]
    rows.extend(edges_to_rows(g.edges(), key_name="scope", key_value="pooled"))

    output_path = paths.results_tables_dir / "case_macro_edges.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    LOGGER.info("Saved case-study edges to %s", output_path)


if __name__ == "__main__":
    run()

