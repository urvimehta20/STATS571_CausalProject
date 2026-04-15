from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class CDNOTSConfig:
    """Algorithm configuration values for CD-NOTS."""

    max_lag: int = 1
    alpha: float = 0.05
    ci_method: str = "parcorr"
    max_condition_set: int = 2


@dataclass
class CDNOTSResult:
    """Typed fit result with compatibility helper for legacy dict usage."""

    graph: nx.DiGraph
    sepsets: Dict[Tuple[str, str], Set[str]]
    lagged_data: pd.DataFrame
    ci_stability_summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Return legacy dictionary output for backward compatibility."""
        return {
            "graph": self.graph,
            "sepsets": self.sepsets,
            "lagged_data": self.lagged_data,
            "ci_stability_summary": self.ci_stability_summary,
        }
