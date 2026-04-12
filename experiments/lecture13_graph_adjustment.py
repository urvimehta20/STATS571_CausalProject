"""
Lecture 13-style adjustment using the *directed* part of a CD-NOD graph.

Reads `discovery2/outputs/cdnod_<tag>_directed_edges.csv` (from run_causal_learn.py),
builds a simple adjustment set L = {parents of Z in that digraph}, then runs

    Y ~ Z + L

with HAC standard errors (time series–friendly; aligns with Lectures 15–16 spirit).

This is a *heuristic*: CD-NOD can return a PDAG; we only use edges classified as
fully directed. Undirected adjacencies are ignored here—see Dagitty / manual
backdoor reasoning for a full Markov equivalence class analysis.

Usage (repo root, after regenerating discovery2 outputs):

    python -m experiments.lecture13_graph_adjustment --tag famafrench --z SMB --y HML
    # RMW -> Mkt_RF: L_graph = parents(RMW) = {SMB, HML} per directed CD-NOD output
    python -m experiments.lecture13_graph_adjustment --tag famafrench --z RMW --y Mkt_RF
    python -m experiments.lecture13_graph_adjustment --tag macro_US \\
        --z unemployment --y cpi --country US --cpi-diff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def _parents_z(directed: pd.DataFrame, z: str) -> list[str]:
    mask = directed["to"].astype(str) == z
    parents = sorted(directed.loc[mask, "from"].astype(str).unique().tolist())
    return parents


def _load_famafrench(root: Path) -> pd.DataFrame:
    p = root / "data" / "raw" / "famafrench_apple_daily.csv"
    return pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def _load_macro(root: Path, country: str | None, cpi_diff: bool) -> pd.DataFrame:
    p = root / "data" / "raw" / "macro_countries_monthly.csv"
    df = pd.read_csv(p, parse_dates=["date"]).sort_values(["country", "date"])
    if country:
        df = df[df["country"] == country].copy()
    if cpi_diff:
        df = df.sort_values(["country", "date"])
        df["cpi"] = df.groupby("country", sort=False)["cpi"].diff()
        df = df.dropna(subset=["cpi"])
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lecture 13-style OLS using CD-NOD directed edges.")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--tag", required=True, help="e.g. famafrench, macro_US, macro_all")
    parser.add_argument("--z", required=True, help="Treatment column name in the data CSV")
    parser.add_argument("--y", required=True, help="Outcome column name")
    parser.add_argument("--country", default=None, help="Macro only: filter to this country")
    parser.add_argument("--cpi-diff", action="store_true", help="Macro: use first difference of CPI as Y")
    parser.add_argument("--lag-z", type=int, default=0, help="Shift Z back this many rows (e.g. 1)")
    parser.add_argument("--hac-lags", type=int, default=5)
    parser.add_argument("--extra-controls", nargs="*", default=[], help="Additional regressors (always included)")
    args = parser.parse_args()

    root = args.project_root.resolve()
    dig_path = root / "discovery2" / "outputs" / f"cdnod_{args.tag}_directed_edges.csv"
    if not dig_path.is_file():
        raise FileNotFoundError(
            f"Missing {dig_path}. Run: python discovery2/run_causal_learn.py --dataset ... "
            "then use matching --tag (e.g. famafrench or macro_US)."
        )
    directed = pd.read_csv(dig_path)

    if args.tag.startswith("macro"):
        df = _load_macro(root, args.country, args.cpi_diff)
    else:
        df = _load_famafrench(root)

    z_col = args.z
    y_col = args.y
    if z_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Z={z_col} or Y={y_col} not in data columns: {list(df.columns)}")

    parents = _parents_z(directed, z_col)
    skip = {z_col, y_col, "Date", "date", "country"}
    L = [p for p in parents if p not in skip and p in df.columns]
    missing_parents = [p for p in parents if p not in df.columns and p not in skip]
    if missing_parents:
        print(f"Note: parents of Z not in data (skipped): {missing_parents}")

    extras = [c for c in args.extra_controls if c in df.columns]
    regressors = [z_col] + L + extras
    use = df[[y_col] + regressors].dropna()

    if args.lag_z:
        use = use.copy()
        use[z_col] = use[z_col].shift(args.lag_z)
        use = use.dropna()

    X = sm.add_constant(use[regressors])
    y = use[y_col]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": args.hac_lags})

    print("=== Lecture 13 heuristic: backdoor adjustment via parents(Z) in directed CD-NOD subgraph ===")
    print(f"Tag={args.tag}  Z={z_col}" + (f" (lag {args.lag_z})" if args.lag_z else "") + f"  Y={y_col}")
    print(f"Adjustment set L (from graph) = {L}")
    if extras:
        print(f"Extra controls (user) = {extras}")
    print(model.summary())

    out = root / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    row = {
        "tag": args.tag,
        "z": z_col,
        "y": y_col,
        "lag_z": args.lag_z,
        "L_graph": ";".join(L),
        "L_extra": ";".join(extras),
        "coef_z": float(model.params[z_col]),
        "se_z": float(model.bse[z_col]),
        "pvalue_z": float(model.pvalues[z_col]),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
    }
    pd.DataFrame([row]).to_csv(out / f"lecture13_adjust_{args.tag}_{z_col}_{y_col}.csv", index=False)
    print(f"\nSaved: {out / f'lecture13_adjust_{args.tag}_{z_col}_{y_col}.csv'}")


if __name__ == "__main__":
    main()
