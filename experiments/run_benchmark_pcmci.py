from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.cdnots.core import CDNOTS, CDNOTSConfig
from src.cdnots.metrics import precision_recall_f1

try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
except Exception:  # pragma: no cover
    pp = None
    PCMCI = None
    ParCorr = None


def generate_data(seed: int = 42, n_obs: int = 500):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n_obs)
    x1 = 0.6 * np.roll(x0, 1) + rng.normal(scale=0.7, size=n_obs)
    x2 = -0.5 * np.roll(x1, 1) + rng.normal(scale=0.8, size=n_obs)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2}).iloc[2:].reset_index(drop=True)


def run():
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_data()
    true_edges = {("X0", "X1"), ("X1", "X2")}

    cd = CDNOTS(CDNOTSConfig(max_lag=1, ci_method="parcorr")).fit(df)["graph"]
    cd_edges = {(u.split("_t")[0], v.split("_t")[0]) for u, v in cd.edges() if "_t-" not in u and "_t-" not in v}
    cd_metrics = precision_recall_f1(cd_edges, true_edges)

    rows = [{"method": "CD-NOTS-ParCorr", **cd_metrics}]
    if PCMCI is not None:
        tdata = pp.DataFrame(df.values, var_names=list(df.columns))
        pcmci = PCMCI(dataframe=tdata, cond_ind_test=ParCorr(significance="analytic"))
        res = pcmci.run_pcmci(tau_max=1, pc_alpha=0.05)
        q = res.get("q_matrix", res.get("p_matrix"))
        if q is None:
            raise RuntimeError("PCMCI output did not contain q_matrix or p_matrix.")
        pcmci_edges = set()
        for i, s in enumerate(df.columns):
            for j, t in enumerate(df.columns):
                if i != j and q[i, j, 1] < 0.05:
                    pcmci_edges.add((s, t))
        rows.append({"method": "PCMCI-ParCorr", **precision_recall_f1(pcmci_edges, true_edges)})

    pd.DataFrame(rows).to_csv(out_dir / "benchmark_pcmci.csv", index=False)


if __name__ == "__main__":
    run()

