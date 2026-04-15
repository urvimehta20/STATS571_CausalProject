from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .graph_decode import save_fallback_png
from .services import DiscoveryArtifacts


class ArtifactWriter:
    """Persist discovery outputs with a stable file contract."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_all(self, run_name: str, artifacts: DiscoveryArtifacts) -> None:
        base = self.output_dir / f"cdnod_{run_name}"
        base.with_suffix(".dot").write_text(artifacts.dot_graph, encoding="utf-8")
        artifacts.edges.to_csv(base.with_name(base.name + "_edges.csv"), index=False)
        pd.DataFrame(artifacts.directed_edges, columns=["from", "to"]).to_csv(
            base.with_name(base.name + "_directed_edges.csv"),
            index=False,
        )
        pd.DataFrame(artifacts.undirected_edges, columns=["node_a", "node_b"]).to_csv(
            base.with_name(base.name + "_undirected_edges.csv"),
            index=False,
        )
        pd.DataFrame(
            {
                "node": artifacts.node_names,
                "is_context": [idx == len(artifacts.node_names) - 1 for idx in range(len(artifacts.node_names))],
            }
        ).to_csv(base.with_name(base.name + "_nodes.csv"), index=False)
        self._render_graph(base, artifacts)

    @staticmethod
    def _render_graph(base_path: Path, artifacts: DiscoveryArtifacts) -> None:
        if shutil.which("dot"):
            import pydot

            graph_list = pydot.graph_from_dot_data(artifacts.dot_graph)
            if not graph_list:
                raise RuntimeError("Could not parse generated DOT graph.")
            graph = graph_list[0]
            graph.write_png(str(base_path.with_suffix(".png")))
            graph.write_pdf(str(base_path.with_suffix(".pdf")))
            return
        save_fallback_png(artifacts.adjacency, artifacts.node_names, str(base_path.with_suffix(".png")))
