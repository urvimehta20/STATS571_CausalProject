from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LagNode:
    name: str
    lag: int

    def key(self) -> str:
        return f"{self.name}_t-{self.lag}" if self.lag > 0 else f"{self.name}_t"


def build_lagged_frame(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    out = []
    for lag in range(max_lag + 1):
        shifted = df.shift(lag).copy()
        shifted.columns = [f"{c}_t-{lag}" if lag > 0 else f"{c}_t" for c in df.columns]
        out.append(shifted)
    merged = pd.concat(out, axis=1).dropna().reset_index(drop=True)
    merged["T"] = np.arange(len(merged))
    return merged


def powerset_limited(items: Sequence[str], k_max: int) -> Iterable[Tuple[str, ...]]:
    yield tuple()
    for k in range(1, min(k_max, len(items)) + 1):
        for c in combinations(items, k):
            yield c


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

