from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

_MAX_ABS_Z = 10.0
_RIDGE_EPS = 1e-6
_MAX_ABS_COEF = 1e6


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
        self.stats = {
            "tests_run": 0,
            "insufficient_rows": 0,
            "degenerate_pearson": 0,
            "nan_result": 0,
        }

    def test(self, data: pd.DataFrame, x: str, y: str, z: Optional[Sequence[str]] = None) -> CITestResult:
        self.stats["tests_run"] += 1
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
            r, p = self._safe_pearsonr(xv, yv)
            if np.isnan(p):
                p, r = 1.0, 0.0
            return CITestResult(float(p), float(r), "parcorr")

        zmat = np.column_stack([self._safe_standardize(data[c].to_numpy()) for c in z])
        zmat = np.column_stack([np.ones(len(zmat), dtype=float), zmat])
        keep = np.isfinite(xv) & np.isfinite(yv) & np.all(np.isfinite(zmat), axis=1)
        if keep.sum() < 3:
            self.stats["insufficient_rows"] += 1
            return CITestResult(1.0, 0.0, "parcorr")
        xv = xv[keep]
        yv = yv[keep]
        zmat = zmat[keep]
        bx = self._ridge_beta(zmat, xv)
        by = self._ridge_beta(zmat, yv)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pred_x = zmat @ bx
            pred_y = zmat @ by
        rx = np.nan_to_num(xv - pred_x, nan=0.0, posinf=0.0, neginf=0.0)
        ry = np.nan_to_num(yv - pred_y, nan=0.0, posinf=0.0, neginf=0.0)
        r, p = self._safe_pearsonr(rx, ry)
        if np.isnan(p):
            self.stats["nan_result"] += 1
            p, r = 1.0, 0.0
        return CITestResult(float(p), float(r), "parcorr")

    @staticmethod
    def _safe_standardize(arr: np.ndarray) -> np.ndarray:
        v = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = float(np.max(np.abs(v))) if v.size else 0.0
        if max_abs > 1e6 and np.isfinite(max_abs):
            v = v / max_abs
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            mean = v.mean() if v.size else 0.0
            std = v.std() if v.size else 0.0
        if (not np.isfinite(std)) or std == 0.0:
            return np.zeros_like(v)
        z = (v - mean) / std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(z, -_MAX_ABS_Z, _MAX_ABS_Z)

    def _safe_pearsonr(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        if a.size < 3 or b.size < 3:
            self.stats["degenerate_pearson"] += 1
            return 0.0, 1.0
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            self.stats["degenerate_pearson"] += 1
            return 0.0, 1.0
        with np.errstate(all="ignore"):
            r, p = stats.pearsonr(a, b)
        if not np.isfinite(r) or not np.isfinite(p):
            self.stats["degenerate_pearson"] += 1
            return 0.0, 1.0
        return float(r), float(p)

    @staticmethod
    def _ridge_beta(zmat: np.ndarray, target: np.ndarray) -> np.ndarray:
        n_cols = zmat.shape[1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ztz = zmat.T @ zmat
            zty = zmat.T @ target
        trace = float(np.trace(ztz)) if np.isfinite(np.trace(ztz)) else 0.0
        lam = _RIDGE_EPS * (trace / max(n_cols, 1) + 1.0)
        reg = ztz + lam * np.eye(n_cols, dtype=float)
        try:
            beta = np.linalg.solve(reg, zty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(zmat, target, rcond=None)[0]
        beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(beta, -_MAX_ABS_COEF, _MAX_ABS_COEF)

    def _fallback_to_parcorr(self, data: pd.DataFrame, x: str, y: str, z: Sequence[str], method: str) -> CITestResult:
        result = self._parcorr(data, x, y, z)
        result.method = f"{method}_fallback_parcorr"
        return result

    def get_stability_summary(self) -> dict[str, int]:
        return dict(self.stats)

