"""
Class-oriented adjustment using the *directed* part of a CD-NOD graph.

Reads `discovery2/outputs/cdnod_<tag>_directed_edges.csv` (from run_causal_learn.py),
builds a simple adjustment set L = {parents of Z in that digraph}, then runs

    Y ~ Z + L

with HAC standard errors. This follows the course framing of graph-based
backdoor adjustment plus outcome regression.

This is a *heuristic*: CD-NOD can return a PDAG; we only use edges classified as
fully directed. Undirected adjacencies are ignored here—see Dagitty / manual
backdoor reasoning for a full Markov equivalence class analysis.

Usage (repo root, after regenerating discovery2 outputs):

    python -m experiments.class_oriented_graph_adjustment --tag famafrench --z SMB --y HML
    # RMW -> Mkt_RF: L_graph = parents(RMW) = {SMB, HML} per directed CD-NOD output
    python -m experiments.class_oriented_graph_adjustment --tag famafrench --z RMW --y Mkt_RF
    python -m experiments.class_oriented_graph_adjustment --tag macro_US \\
        --z unemployment --y cpi --country US --cpi-diff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from experiments.causal_diagnostics import candidate_controls_from_graph, graph_ambiguity_flag
from experiments.common import build_paths, load_famafrench_daily, load_macro_monthly
from src.cdnots.project_io import get_logger

LOGGER = get_logger("experiments.class_oriented_graph_adjustment")


def _parents_z(directed: pd.DataFrame, z: str) -> list[str]:
    mask = directed["to"].astype(str) == z
    parents = sorted(directed.loc[mask, "from"].astype(str).unique().tolist())
    return parents


def _load_macro(paths_root: Path, country: str | None, cpi_diff: bool) -> pd.DataFrame:
    paths = build_paths(paths_root)
    df = load_macro_monthly(paths).sort_values(["country", "date"])
    if country:
        df = df[df["country"] == country].copy()
    if cpi_diff:
        df = df.sort_values(["country", "date"])
        df["cpi"] = df.groupby("country", sort=False)["cpi"].diff()
        df = df.dropna(subset=["cpi"])
    return df.reset_index(drop=True)


def _hidden_confounding_sensitivity(t_stat: float, dof: float) -> tuple[float, float]:
    """
    Return a simple robustness proxy using partial R^2 style calculations.
    Values closer to 1 imply stronger residualized treatment-outcome signal.
    """
    if dof <= 0:
        return 0.0, 0.0
    t_sq = float(t_stat) ** 2
    partial_r2 = t_sq / (t_sq + dof) if (t_sq + dof) > 0 else 0.0
    robustness_value = partial_r2 / (1.0 - partial_r2 + 1e-9)
    return float(partial_r2), float(robustness_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-oriented OLS using CD-NOD directed edges.")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--tag", required=True, help="e.g. famafrench, macro_US, macro_all")
    parser.add_argument("--z", required=True, help="Treatment column name in the data CSV")
    parser.add_argument("--y", required=True, help="Outcome column name")
    parser.add_argument("--country", default=None, help="Macro only: filter to this country")
    parser.add_argument("--cpi-diff", action="store_true", help="Macro: use first difference of CPI as Y")
    parser.add_argument("--lag-z", type=int, default=0, help="Shift Z back this many rows (e.g. 1)")
    parser.add_argument("--hac-lags", type=int, default=5)
    parser.add_argument(
        "--adjustment-mode",
        choices=["parents", "minimal_backdoor"],
        default="parents",
        help="Adjustment candidate mode from graph structure.",
    )
    parser.add_argument("--extra-controls", nargs="*", default=[], help="Additional regressors (always included)")
    args = parser.parse_args()

    root = args.project_root.resolve()
    paths = build_paths(root)
    dig_path = root / "discovery2" / "outputs" / f"cdnod_{args.tag}_directed_edges.csv"
    if not dig_path.is_file():
        raise FileNotFoundError(
            f"Missing {dig_path}. Run: python discovery2/run_causal_learn.py --dataset ... "
            "then use matching --tag (e.g. famafrench or macro_US)."
        )
    directed = pd.read_csv(dig_path)
    undig_path = root / "discovery2" / "outputs" / f"cdnod_{args.tag}_undirected_edges.csv"
    undirected = pd.read_csv(undig_path) if undig_path.is_file() else None

    if args.tag.startswith("macro"):
        df = _load_macro(root, args.country, args.cpi_diff)
    else:
        df = load_famafrench_daily(paths)

    z_col = args.z
    y_col = args.y
    if z_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Z={z_col} or Y={y_col} not in data columns: {list(df.columns)}")

    parent_controls = _parents_z(directed, z_col)
    graph_controls = candidate_controls_from_graph(
        treatment=z_col,
        outcome=y_col,
        directed_edges=directed,
        mode=args.adjustment_mode,
        available_columns=df.columns,
    )
    skip = {z_col, y_col, "Date", "date", "country"}
    missing_parents = [p for p in parent_controls if p not in df.columns and p not in skip]
    if missing_parents:
        LOGGER.warning("Parents of Z missing in data and skipped: %s", missing_parents)

    extras = [c for c in args.extra_controls if c in df.columns]
    regressors = [z_col] + graph_controls + [c for c in extras if c not in graph_controls and c != z_col]
    use = df[[y_col] + regressors].dropna()

    if args.lag_z:
        use = use.copy()
        use[z_col] = use[z_col].shift(args.lag_z)
        use = use.dropna()
    if use.empty:
        raise ValueError("No usable rows remain after filtering/lagging; adjust inputs or lag settings.")

    X = sm.add_constant(use[regressors])
    y = use[y_col]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": args.hac_lags})
    ambiguity_flag, ambiguity_reason = graph_ambiguity_flag(
        treatment=z_col,
        outcome=y_col,
        undirected_edges=undirected,
        selected_controls=graph_controls,
    )
    t_stat = float(model.tvalues[z_col]) if z_col in model.tvalues else 0.0
    partial_r2, robustness_value = _hidden_confounding_sensitivity(t_stat=t_stat, dof=float(model.df_resid))

    print("=== Class-oriented heuristic: backdoor adjustment via parents(Z) in directed CD-NOD subgraph ===")
    print(f"Tag={args.tag}  Z={z_col}" + (f" (lag {args.lag_z})" if args.lag_z else "") + f"  Y={y_col}")
    print(f"Adjustment mode = {args.adjustment_mode}")
    print(f"Adjustment set L (from graph) = {graph_controls}")
    if ambiguity_flag:
        print(f"Identification ambiguity flag: {ambiguity_reason}")
    if extras:
        print(f"Extra controls (user) = {extras}")
    print(model.summary())

    out = paths.results_tables_dir
    row = {
        "tag": args.tag,
        "z": z_col,
        "y": y_col,
        "lag_z": args.lag_z,
        "adjustment_mode": args.adjustment_mode,
        "L_graph": ";".join(graph_controls),
        "L_extra": ";".join(extras),
        "graph_ambiguity_flag": int(ambiguity_flag),
        "graph_ambiguity_reason": ambiguity_reason,
        "coef_z": float(model.params[z_col]),
        "se_z": float(model.bse[z_col]),
        "pvalue_z": float(model.pvalues[z_col]),
        "t_z": t_stat,
        "partial_r2_z": partial_r2,
        "robustness_value_z": robustness_value,
        "n": int(model.nobs),
        "r2": float(model.rsquared),
    }
    output_path = out / f"class_oriented_adjust_{args.tag}_{z_col}_{y_col}.csv"
    pd.DataFrame([row]).to_csv(output_path, index=False)
    LOGGER.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
