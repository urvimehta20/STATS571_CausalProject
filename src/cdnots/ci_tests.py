from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CITestResult:
    p_value: float
    statistic: float
    method: str


class CITester:
    def __init__(self, method: str = "parcorr", alpha: float = 0.05, random_state: int = 0):
        self.method = method.lower()
        self.alpha = alpha
        self.random_state = random_state

    def test(self, data: pd.DataFrame, x: str, y: str, z: Optional[Sequence[str]] = None) -> CITestResult:
        z = list(z or [])
        if self.method == "parcorr":
            return self._parcorr(data, x, y, z)
        if self.method in {"kcit", "kcit_hbe", "kcit_sw"}:
            return self._fallback_to_parcorr(data, x, y, z, self.method)
        if self.method in {"rcot", "rcot_hbe", "rcot_sw"}:
            return self._fallback_to_parcorr(data, x, y, z, self.method)
        if self.method == "cmiknn":
            return self._fallback_to_parcorr(data, x, y, z, self.method)
        raise ValueError(f"Unsupported CI method: {self.method}")

    def _parcorr(self, data: pd.DataFrame, x: str, y: str, z: Sequence[str]) -> CITestResult:
        xv = self._safe_standardize(data[x].to_numpy())
        yv = self._safe_standardize(data[y].to_numpy())
        if not z:
            r, p = stats.pearsonr(xv, yv)
            if np.isnan(p):
                p, r = 1.0, 0.0
            return CITestResult(float(p), float(r), "parcorr")

        zmat = np.column_stack([self._safe_standardize(data[c].to_numpy()) for c in z])
        zmat = np.column_stack([np.ones(len(zmat)), zmat])
        bx = np.linalg.pinv(zmat) @ xv
        by = np.linalg.pinv(zmat) @ yv
        rx = xv - zmat @ bx
        ry = yv - zmat @ by
        r, p = stats.pearsonr(rx, ry)
        if np.isnan(p):
            p, r = 1.0, 0.0
        return CITestResult(float(p), float(r), "parcorr")

    @staticmethod
    def _safe_standardize(arr: np.ndarray) -> np.ndarray:
        v = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        std = v.std()
        if std == 0.0:
            return np.zeros_like(v)
        return (v - v.mean()) / std

    def _fallback_to_parcorr(self, data: pd.DataFrame, x: str, y: str, z: Sequence[str], method: str) -> CITestResult:
        result = self._parcorr(data, x, y, z)
        result.method = f"{method}_fallback_parcorr"
        return result

