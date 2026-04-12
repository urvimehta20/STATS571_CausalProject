from __future__ import annotations

import argparse
import os
import shutil
import warnings
from pathlib import Path
from typing import Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.GraphUtils import GraphUtils


def _prepare_famafrench(raw_path: Path, max_rows: int | None) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    if max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    feature_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF", "AAPL_RET"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    out = df[feature_cols].copy()

    # CD-NOD expects a context variable index. We use time index as context
    out = out.dropna().reset_index(drop=True)
    c_indx = np.arange(len(out), dtype=float).reshape(-1, 1)
    data = out.to_numpy(dtype=float)
    return data, c_indx, out.columns.tolist() + ["context_time"]


def _prepare_macro(raw_path: Path, max_rows: int | None, country: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(raw_path, parse_dates=["date"]).sort_values(["country", "date"]).reset_index(drop=True)
    if country.lower() != "all":
        df = df[df["country"] == country].copy()
    if max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    numeric = df[["unemployment", "cpi", "ppi"]].astype(float)
    valid_mask = numeric.notna().all(axis=1)
    numeric = numeric.loc[valid_mask]
    countries = df.loc[valid_mask, "country"]
    # Stable integer context for country identity
    out = numeric.reset_index(drop=True)
    country_codes = pd.Categorical(countries.reset_index(drop=True)).codes.astype(float).reshape(-1, 1)
    data = out.to_numpy(dtype=float)
    return data, country_codes, out.columns.tolist() + ["context_country"]


def _edges_from_adj(adj: np.ndarray, node_names: list[str]) -> pd.DataFrame:
    rows = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i, j] != 0:
                rows.append({"from": node_names[i], "to": node_names[j], "weight": float(adj[i, j])})
    return pd.DataFrame(rows)


def _decompose_adj_for_plot(adj: np.ndarray, node_names: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    causal-learn GeneralGraph matrix (Tetrad-style):
    - Directed i --> j: adj[i,j] == -1 (tail at j) and adj[j,i] == 1 (arrow at i), i.e. is_fully_directed(i,j).
    - Undirected i --- j: adj[i,j] == adj[j,i] == -1
    - Bidirected i <-> j: adj[i,j] == adj[j,i] == 1  (draw as undirected for clarity)
    - Other PDAG endpoints (circle, etc.): draw as undirected
    """
    n = adj.shape[0]
    directed: list[tuple[str, str]] = []
    undirected: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = float(adj[i, j]), float(adj[j, i])
            if a == 0 and b == 0:
                continue
            u, v = node_names[i], node_names[j]
            if a == -1 and b == 1:
                directed.append((u, v))
            elif a == 1 and b == -1:
                directed.append((v, u))
            elif a == -1 and b == -1:
                undirected.append((u, v))
            elif a == 1 and b == 1:
                undirected.append((u, v))
            else:
                undirected.append((u, v))
    return directed, undirected


def _save_fallback_png(adj: np.ndarray, node_names: list[str], out_png: Path) -> None:
    directed, undirected = _decompose_adj_for_plot(adj, node_names)
    nodes = list(node_names)
    g_layout = nx.Graph()
    g_layout.add_nodes_from(nodes)
    for u, v in undirected:
        g_layout.add_edge(u, v)
    for u, v in directed:
        g_layout.add_edge(u, v)
    if g_layout.number_of_edges() == 0:
        g_layout.add_nodes_from(nodes)
    pos = nx.spring_layout(g_layout, seed=42, k=2.0 / max(np.sqrt(len(nodes)), 1))

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    nx.draw_networkx_nodes(
        g_layout, pos, nodelist=nodes, node_size=1400, node_color="#E8F1FF", edgecolors="#335C99", ax=ax
    )
    nx.draw_networkx_labels(g_layout, pos, labels={n: n for n in nodes}, font_size=9, ax=ax)

    if undirected:
        ug = nx.Graph()
        ug.add_edges_from(undirected)
        nx.draw_networkx_edges(
            ug,
            pos,
            edgelist=list(ug.edges()),
            width=2.2,
            edge_color="#6B6B6B",
            style="solid",
            arrows=False,
            ax=ax,
        )

    if directed:
        dg = nx.DiGraph()
        dg.add_edges_from(directed)
        nx.draw_networkx_edges(
            dg,
            pos,
            edgelist=list(dg.edges()),
            width=2.0,
            edge_color="#1a1a1a",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            connectionstyle="arc3,rad=0.08",
            min_source_margin=18,
            min_target_margin=18,
            ax=ax,
        )

    plt.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def run(args: argparse.Namespace) -> None:
    root = Path(args.project_root).resolve()
    out_dir = (root / "discovery2" / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "famafrench":
        raw_path = root / "data" / "raw" / "famafrench_apple_daily.csv"
        data, c_indx, node_names = _prepare_famafrench(raw_path, args.max_rows)
        run_name = "famafrench"
    else:
        raw_path = root / "data" / "raw" / "macro_countries_monthly.csv"
        data, c_indx, node_names = _prepare_macro(raw_path, args.max_rows, args.country)
        run_name = f"macro_{args.country}"

    if data.shape[0] < 10:
        raise ValueError(f"Not enough rows after preprocessing: {data.shape[0]}")

    print(f"Running CD-NOD on {run_name} with shape={data.shape}, c_indx_shape={c_indx.shape}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        cg = cdnod(data, c_indx, alpha=args.alpha)

    pyd = GraphUtils.to_pydot(cg.G, labels=node_names)
    base = out_dir / f"cdnod_{run_name}"
    dot_path = base.with_suffix(".dot")
    dot_path.write_text(pyd.to_string(), encoding="utf-8")

    adj = np.asarray(cg.G.graph)
    edges = _edges_from_adj(adj, node_names)
    edges.to_csv(base.with_name(base.name + "_edges.csv"), index=False)

    directed, undirected = _decompose_adj_for_plot(adj, node_names)
    pd.DataFrame(directed, columns=["from", "to"]).to_csv(
        base.with_name(base.name + "_directed_edges.csv"), index=False
    )
    pd.DataFrame(undirected, columns=["node_a", "node_b"]).to_csv(
        base.with_name(base.name + "_undirected_edges.csv"), index=False
    )

    if shutil.which("dot"):
        pyd.write_png(str(base.with_suffix(".png")))
        pyd.write_pdf(str(base.with_suffix(".pdf")))
    else:
        _save_fallback_png(adj, node_names, base.with_suffix(".png"))
        print("Graphviz 'dot' not found. Saved PNG via NetworkX fallback; skipped PDF.")

    pd.DataFrame({"node": node_names, "is_context": [i == len(node_names) - 1 for i in range(len(node_names))]}).to_csv(
        base.with_name(base.name + "_nodes.csv"), index=False
    )
    print(f"Saved graph files under: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run causal-learn CD-NOD on raw datasets.")
    parser.add_argument("--project-root", default=".", help="Repository root path (default: current directory).")
    parser.add_argument("--dataset", choices=["famafrench", "macro"], required=True)
    parser.add_argument("--country", default="US", help="Macro country code (or 'all').")
    parser.add_argument("--alpha", type=float, default=0.05, help="CD-NOD alpha level.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap to last N rows.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
