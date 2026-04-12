"""
Estimate the association / lag-predictive relationship of SMB on HML using
Fama-French + Apple daily data.

CD-NOD output (discovery2) does not show a direct SMB–HML edge; both connect via RMW.
These regressions are for exploration; contemporaneous SMB is not a causal treatment
without stronger structural assumptions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "raw" / "famafrench_apple_daily.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    cols = ["SMB", "HML", "Mkt_RF", "RMW", "CMA"]
    df = df.dropna(subset=cols)

    # Spec A: SMB known before day's HML (lagged SMB → HML_t)
    df["SMB_lag1"] = df["SMB"].shift(1)
    d_lag = df.dropna(subset=["SMB_lag1"])
    y_a = d_lag["HML"]
    X_a = sm.add_constant(d_lag[["SMB_lag1", "Mkt_RF", "RMW", "CMA"]])
    m_lag = sm.OLS(y_a, X_a).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    # Spec B: contemporaneous (associational)
    y_b = df["HML"]
    X_b = sm.add_constant(df[["SMB", "Mkt_RF", "RMW", "CMA"]])
    m_contemp = sm.OLS(y_b, X_b).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    print("=== HML ~ SMB_lag1 + Mkt_RF + RMW + CMA (Newey–West HAC, maxlags=5) ===")
    print(m_lag.summary())
    print("\n=== HML ~ SMB + same controls, same day (associational) ===")
    print(m_contemp.summary())

    out = root / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "spec": "lagged_SMB",
            "coef_SMB": m_lag.params.get("SMB_lag1", float("nan")),
            "se_SMB": m_lag.bse.get("SMB_lag1", float("nan")),
            "pvalue_SMB": m_lag.pvalues.get("SMB_lag1", float("nan")),
            "n": int(m_lag.nobs),
            "r2": float(m_lag.rsquared),
        },
        {
            "spec": "contemp_SMB",
            "coef_SMB": m_contemp.params.get("SMB", float("nan")),
            "se_SMB": m_contemp.bse.get("SMB", float("nan")),
            "pvalue_SMB": m_contemp.pvalues.get("SMB", float("nan")),
            "n": int(m_contemp.nobs),
            "r2": float(m_contemp.rsquared),
        },
    ]
    pd.DataFrame(rows).to_csv(out / "effect_smb_hml.csv", index=False)
    print(f"\nSaved summary: {out / 'effect_smb_hml.csv'}")


if __name__ == "__main__":
    main()
