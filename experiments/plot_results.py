from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    sim = Path("results/tables/simulation_summary.csv")
    if sim.exists():
        df = pd.read_csv(sim)
        plt.figure(figsize=(9, 5))
        for (method, n_obs), sub in df.groupby(["ci_method", "n_obs"]):
            plt.plot(sub["n_nodes"], sub["f1"], marker="o", label=f"{method} (n={n_obs})")
        plt.ylim(0, 1)
        plt.xlabel("Nodes")
        plt.ylabel("F1")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / "simulation_f1.png", dpi=150)
        plt.close()

    ff_path = Path("data/raw/famafrench_apple_daily.csv")
    if ff_path.exists():
        ff = pd.read_csv(ff_path, parse_dates=["Date"]).set_index("Date")
        cols = [c for c in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "AAPL_RET"] if c in ff.columns]
        if cols:
            ff[cols].plot(subplots=True, figsize=(10, 9), legend=False)
            plt.tight_layout()
            plt.savefig(fig_dir / "case_famafrench_apple_series.png", dpi=150)
            plt.close("all")


if __name__ == "__main__":
    main()

