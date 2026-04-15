from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .ci_tests import CITester
from .models import CDNOTSConfig, CDNOTSResult
from .stages import OrientationStage, SkeletonDiscoveryStage
from .utils import build_lagged_frame


class CDNOTS:
    """Public CD-NOTS API with legacy dict output compatibility."""

    def __init__(self, config: Optional[CDNOTSConfig] = None):
        self.config = config or CDNOTSConfig()
        self.ci = CITester(method=self.config.ci_method, alpha=self.config.alpha)
        self.skeleton_stage = SkeletonDiscoveryStage(
            ci_tester=self.ci,
            alpha=self.config.alpha,
            max_condition_set=self.config.max_condition_set,
        )
        self.orientation_stage = OrientationStage()

    def fit(self, df: pd.DataFrame) -> Dict[str, object]:
        """Fit CD-NOTS and return legacy dict-compatible result."""
        return self.fit_result(df).to_dict()

    def fit_result(self, df: pd.DataFrame) -> CDNOTSResult:
        """Fit CD-NOTS and return a typed result object."""
        lagged = build_lagged_frame(df, self.config.max_lag)
        graph, sepsets = self.skeleton_stage.run(lagged)
        graph = self.orientation_stage.run_stage3(graph, sepsets)
        graph = self.orientation_stage.run_stage4(graph)
        return CDNOTSResult(
            graph=graph,
            sepsets=sepsets,
            lagged_data=lagged,
            ci_stability_summary=self.ci.get_stability_summary(),
        )

