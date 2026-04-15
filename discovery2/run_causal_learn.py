from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")

from discovery2.io_utils import ArtifactWriter
from discovery2.services import CDNODPipeline, DatasetPreprocessor


def run(args: argparse.Namespace) -> None:
    root = Path(args.project_root).resolve()
    if args.dataset == "famafrench":
        raw_path = root / "data" / "raw" / "famafrench_apple_daily.csv"
        prepared = DatasetPreprocessor.prepare_famafrench(raw_path, args.max_rows)
    else:
        raw_path = root / "data" / "raw" / "macro_countries_monthly.csv"
        prepared = DatasetPreprocessor.prepare_macro(raw_path, args.max_rows, args.country)

    print(
        f"Running CD-NOD on {prepared.run_name} with "
        f"shape={prepared.data.shape}, c_indx_shape={prepared.context_index.shape}"
    )
    artifacts = CDNODPipeline(alpha=args.alpha).run(prepared)
    ArtifactWriter(root / "discovery2" / "outputs").write_all(prepared.run_name, artifacts)
    print(f"Saved graph files under: {root / 'discovery2' / 'outputs'}")


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
