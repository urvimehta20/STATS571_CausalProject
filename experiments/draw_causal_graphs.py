from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _lag_level(node: str) -> int:
    if "_t-" in node:
        try:
            return int(node.split("_t-")[-1])
        except ValueError:
            return 0
    if node.endswith("_t"):
        return 0
    return 0


def _base_name(node: str) -> str:
    if "_t-" in node:
        return node.split("_t-")[0]
    if node.endswith("_t"):
        return node[:-2]
    return node


def _build_positions(nodes: list[str]) -> dict[str, tuple[float, float]]:
    lag_levels = sorted({_lag_level(n) for n in nodes})
    lag_to_x = {lag: i for i, lag in enumerate(lag_levels)}

    bases = sorted({_base_name(n) for n in nodes})
    base_to_y = {b: i for i, b in enumerate(bases)}

    pos: dict[str, tuple[float, float]] = {}
    for n in nodes:
        x = lag_to_x[_lag_level(n)]
        y = -base_to_y[_base_name(n)]
        pos[n] = (float(x), float(y))
    return pos


def draw_group_graph(df: pd.DataFrame, title: str, output_path: Path) -> None:
    g = nx.DiGraph()
    for _, row in df.iterrows():
        src = str(row["from"])
        dst = str(row["to"])
        g.add_edge(src, dst)

    if g.number_of_nodes() == 0:
        return

    nodes = sorted(g.nodes())
    pos = _build_positions(nodes)

    plt.figure(figsize=(12, max(5, len({_base_name(n) for n in nodes}) * 0.7)))
    ax = plt.gca()
    ax.set_title(title, fontsize=14, pad=14)

    nx.draw_networkx_nodes(
        g,
        pos,
        node_color="#E8F1FF",
        edgecolors="#335C99",
        node_size=1500,
        linewidths=1.2,
    )

    nx.draw_networkx_labels(g, pos, font_size=9)

    single_dir_edges = []
    bi_dir_edges = []
    seen = set()
    for u, v in g.edges():
        if (u, v) in seen:
            continue
        if g.has_edge(v, u) and u != v:
            bi_dir_edges.append((u, v))
            seen.add((u, v))
            seen.add((v, u))
        else:
            single_dir_edges.append((u, v))
            seen.add((u, v))

    if single_dir_edges:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=single_dir_edges,
            edge_color="#4A4A4A",
            width=1.6,
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.06",
        )

    if bi_dir_edges:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=bi_dir_edges,
            edge_color="#D1495B",
            width=1.6,
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.20",
        )
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[(v, u) for (u, v) in bi_dir_edges],
            edge_color="#D1495B",
            width=1.6,
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.20",
        )

    ax.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw causal graph figures from edge CSV files.")
    parser.add_argument("--input", required=True, help="Path to edge CSV with 'from' and 'to' columns.")
    parser.add_argument(
        "--out-dir",
        default="results/graphs",
        help="Directory where graph files are saved (default: results/graphs).",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Optional grouping column (e.g., period or scope). If omitted, auto-detected.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix. Defaults to input CSV stem.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    df = pd.read_csv(input_path)

    required = {"from", "to"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    group_col = args.group_col
    if group_col is None:
        candidates = [c for c in df.columns if c not in {"from", "to"}]
        group_col = candidates[0] if candidates else None

    prefix = args.prefix or input_path.stem

    if group_col and group_col in df.columns:
        for group_value, sub in df.groupby(group_col):
            tag = _sanitize(group_value)
            out_png = out_dir / f"{prefix}_{tag}.png"
            out_pdf = out_dir / f"{prefix}_{tag}.pdf"
            title = f"{prefix} ({group_col}={group_value})"
            draw_group_graph(sub, title, out_png)
            draw_group_graph(sub, title, out_pdf)
            print(f"Saved: {out_png}")
            print(f"Saved: {out_pdf}")
    else:
        out_png = out_dir / f"{prefix}.png"
        out_pdf = out_dir / f"{prefix}.pdf"
        draw_group_graph(df, prefix, out_png)
        draw_group_graph(df, prefix, out_pdf)
        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
