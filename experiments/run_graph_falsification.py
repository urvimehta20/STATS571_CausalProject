from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import build_paths, load_famafrench_daily, load_macro_monthly
from src.cdnots.ci_tests import CITester
from src.cdnots.project_io import get_logger

LOGGER = get_logger("experiments.run_graph_falsification")


def _load_data(paths_root: Path, tag: str, country: str | None) -> pd.DataFrame:
    paths = build_paths(paths_root)
    if tag.startswith("macro"):
        data = load_macro_monthly(paths)
        if country:
            data = data[data["country"] == country].copy()
        return data.sort_values(["country", "date"]).reset_index(drop=True)
    return load_famafrench_daily(paths)


def _build_parent_map(directed: pd.DataFrame) -> dict[str, set[str]]:
    parent_map: dict[str, set[str]] = {}
    for _, row in directed.iterrows():
        source = str(row["from"])
        target = str(row["to"])
        parent_map.setdefault(target, set()).add(source)
        parent_map.setdefault(source, set())
    return parent_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph implication falsification checks.")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--tag", required=True, help="e.g. famafrench, macro_US, macro_all")
    parser.add_argument("--country", default=None, help="Optional macro country filter")
    parser.add_argument("--ci-method", default="parcorr")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-checks", type=int, default=50)
    args = parser.parse_args()

    root = args.project_root.resolve()
    paths = build_paths(root)
    directed_path = root / "discovery2" / "outputs" / f"cdnod_{args.tag}_directed_edges.csv"
    if not directed_path.is_file():
        raise FileNotFoundError(f"Missing directed edge file: {directed_path}")
    directed = pd.read_csv(directed_path)
    data = _load_data(root, args.tag, args.country)

    adjacency: set[tuple[str, str]] = {
        (str(row["from"]), str(row["to"])) for _, row in directed.iterrows()
    }
    adjacency |= {(target, source) for source, target in adjacency}
    parent_map = _build_parent_map(directed)
    columns = [column for column in data.columns if column not in {"Date", "date", "country"}]
    tester = CITester(method=args.ci_method, alpha=args.alpha)

    records: list[dict[str, object]] = []
    checks = 0
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            if (left, right) in adjacency:
                continue
            conditioning = sorted((parent_map.get(left, set()) | parent_map.get(right, set())) - {left, right})
            conditioning = [name for name in conditioning if name in columns]
            result = tester.test(data, left, right, conditioning)
            records.append(
                {
                    "tag": args.tag,
                    "x": left,
                    "y": right,
                    "conditioning_set": ";".join(conditioning),
                    "p_value": float(result.p_value),
                    "statistic": float(result.statistic),
                    "passes_implied_independence": int(result.p_value > args.alpha),
                    "method": result.method,
                }
            )
            checks += 1
            if checks >= args.max_checks:
                break
        if checks >= args.max_checks:
            break

    result_frame = pd.DataFrame(records)
    output_path = paths.results_tables_dir / f"graph_falsification_{args.tag}.csv"
    result_frame.to_csv(output_path, index=False)
    if not result_frame.empty:
        pass_rate = float(result_frame["passes_implied_independence"].mean())
        LOGGER.info("Saved falsification report to %s (pass_rate=%.3f)", output_path, pass_rate)
    else:
        LOGGER.warning("No falsification checks generated for tag=%s", args.tag)


if __name__ == "__main__":
    main()
