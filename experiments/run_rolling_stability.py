from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from experiments.common import build_paths, load_famafrench_daily, load_macro_monthly
from src.cdnots.core import CDNOTS, CDNOTSConfig
from src.cdnots.project_io import get_logger

LOGGER = get_logger("experiments.run_rolling_stability")


def _load_dataset(paths_root: Path, tag: str, country: str | None) -> pd.DataFrame:
    paths = build_paths(paths_root)
    if tag.startswith("macro"):
        data = load_macro_monthly(paths)
        if country:
            data = data[data["country"] == country].copy()
        data = data.sort_values(["country", "date"]).reset_index(drop=True)
        return data
    return load_famafrench_daily(paths)


def _base_name(node: str) -> str:
    if "_t-" in node:
        return node.split("_t-")[0]
    if node.endswith("_t"):
        return node[:-2]
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-window edge and coefficient stability analysis.")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--tag", default="famafrench")
    parser.add_argument("--country", default=None)
    parser.add_argument("--z", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--max-lag", type=int, default=1)
    parser.add_argument("--ci-method", default="parcorr")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    root = args.project_root.resolve()
    paths = build_paths(root)
    full = _load_dataset(root, args.tag, args.country)
    feature_cols = [column for column in full.columns if column not in {"Date", "date", "country"}]
    if args.z not in feature_cols or args.y not in feature_cols:
        raise ValueError(f"z/y must be in data columns: {feature_cols}")
    full = full[feature_cols].dropna().reset_index(drop=True)
    if len(full) < args.window:
        raise ValueError(f"Need at least {args.window} rows for rolling analysis.")

    rows: list[dict[str, float | int]] = []
    for start in range(0, len(full) - args.window + 1, args.step):
        end = start + args.window
        window_df = full.iloc[start:end].copy()
        model = CDNOTS(
            CDNOTSConfig(max_lag=args.max_lag, ci_method=args.ci_method, alpha=args.alpha, max_condition_set=2)
        )
        graph = model.fit(window_df)["graph"]

        parent_candidates = sorted(
            {
                _base_name(source)
                for source, target in graph.edges()
                if _base_name(target) == args.z and _base_name(source) in window_df.columns and _base_name(source) != args.z
            }
        )
        regressors = [args.z] + [name for name in parent_candidates if name != args.y]
        use = window_df[[args.y] + regressors].dropna()
        if len(use) < 20:
            continue
        fit = sm.OLS(use[args.y], sm.add_constant(use[regressors])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        edge_present = int(any(_base_name(source) == args.z and _base_name(target) == args.y for source, target in graph.edges()))
        rows.append(
            {
                "start": start,
                "end": end,
                "edge_present": edge_present,
                "coef_z": float(fit.params.get(args.z, 0.0)),
                "se_z": float(fit.bse.get(args.z, 0.0)),
                "pvalue_z": float(fit.pvalues.get(args.z, 1.0)),
                "n": int(fit.nobs),
            }
        )

    out = pd.DataFrame(rows)
    table_path = paths.results_tables_dir / f"rolling_stability_{args.tag}_{args.z}_{args.y}.csv"
    out.to_csv(table_path, index=False)
    LOGGER.info("Saved rolling stability table to %s", table_path)

    if not out.empty:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(out["start"], out["coef_z"], marker="o")
        axes[0].set_ylabel("coef_z")
        axes[0].set_title(f"Rolling coefficient stability: {args.z} -> {args.y}")
        axes[1].plot(out["start"], out["edge_present"], marker="o")
        axes[1].set_ylabel("edge_present")
        axes[1].set_xlabel("window_start")
        axes[1].set_yticks([0, 1])
        plt.tight_layout()
        figure_path = paths.results_figures_dir / f"rolling_stability_{args.tag}_{args.z}_{args.y}.png"
        plt.savefig(figure_path, dpi=150)
        plt.close(fig)
        LOGGER.info("Saved rolling stability figure to %s", figure_path)


if __name__ == "__main__":
    main()
