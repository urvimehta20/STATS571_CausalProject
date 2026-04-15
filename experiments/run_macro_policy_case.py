from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from experiments.common import build_paths, load_macro_monthly
from src.cdnots.project_io import get_logger

LOGGER = get_logger("experiments.run_macro_policy_case")


def _synthetic_weights(
    treated_pre: pd.Series,
    donor_pre: pd.DataFrame,
) -> pd.Series:
    if donor_pre.empty:
        return pd.Series(dtype=float)
    solution = np.linalg.lstsq(donor_pre.to_numpy(), treated_pre.to_numpy(), rcond=None)[0]
    weights = np.clip(solution, 0.0, None)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()
    return pd.Series(weights, index=donor_pre.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro policy/event case using DID and synthetic control.")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--treated-country", default="US")
    parser.add_argument("--outcome", choices=["cpi", "ppi", "unemployment"], default="cpi")
    parser.add_argument("--event-date", default="2008-09-01")
    args = parser.parse_args()

    root = args.project_root.resolve()
    paths = build_paths(root)
    macro = load_macro_monthly(paths).copy()
    macro["date"] = pd.to_datetime(macro["date"])
    event_date = pd.to_datetime(args.event_date)
    macro = macro.dropna(subset=[args.outcome])
    macro["treated"] = (macro["country"] == args.treated_country).astype(int)
    macro["post"] = (macro["date"] >= event_date).astype(int)
    macro["treated_post"] = macro["treated"] * macro["post"]

    did_model = smf.ols(
        f"{args.outcome} ~ treated + post + treated_post + C(country) + C(date)",
        data=macro,
    ).fit(cov_type="HC1")
    did_effect = float(did_model.params.get("treated_post", 0.0))
    did_se = float(did_model.bse.get("treated_post", 0.0))
    did_p = float(did_model.pvalues.get("treated_post", 1.0))

    wide = macro.pivot_table(index="date", columns="country", values=args.outcome).sort_index()
    if args.treated_country not in wide.columns:
        raise ValueError(f"Treated country {args.treated_country} not in panel columns.")
    pre_mask = wide.index < event_date
    treated_pre = wide.loc[pre_mask, args.treated_country].dropna()
    donor_pre = wide.loc[treated_pre.index].drop(columns=[args.treated_country]).dropna(axis=1)
    treated_pre = treated_pre.loc[donor_pre.index]
    weights = _synthetic_weights(treated_pre, donor_pre)

    synthetic_series = (wide[weights.index] * weights.values).sum(axis=1) if not weights.empty else pd.Series(index=wide.index, dtype=float)
    treated_series = wide[args.treated_country]
    scm_gap = treated_series - synthetic_series
    scm_effect_post = float(scm_gap[wide.index >= event_date].mean()) if not scm_gap.empty else float("nan")

    result = pd.DataFrame(
        [
            {
                "treated_country": args.treated_country,
                "outcome": args.outcome,
                "event_date": event_date.date().isoformat(),
                "did_effect": did_effect,
                "did_se": did_se,
                "did_pvalue": did_p,
                "scm_avg_gap_post": scm_effect_post,
                "n_countries": int(macro["country"].nunique()),
            }
        ]
    )
    table_path = paths.results_tables_dir / f"macro_policy_case_{args.treated_country}_{args.outcome}.csv"
    result.to_csv(table_path, index=False)
    LOGGER.info("Saved macro policy case table to %s", table_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(treated_series.index, treated_series.values, label=f"Treated: {args.treated_country}")
    if not synthetic_series.empty:
        ax.plot(synthetic_series.index, synthetic_series.values, label="Synthetic control")
    ax.axvline(event_date, linestyle="--", color="black", linewidth=1.0, label="Event date")
    ax.set_title(f"Macro policy case: {args.outcome}")
    ax.legend()
    plt.tight_layout()
    figure_path = paths.results_figures_dir / f"macro_policy_case_{args.treated_country}_{args.outcome}.png"
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved macro policy figure to %s", figure_path)


if __name__ == "__main__":
    main()
