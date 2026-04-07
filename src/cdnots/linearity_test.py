from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF
from scipy.stats import pearsonr


def linearity_test_with_gp(data: pd.DataFrame, x: str, y: str, z: Sequence[str]) -> float:
    z = list(z)
    if z:
        x_mat = data[[x] + z].to_numpy()
        kernel = DotProduct() + RBF(length_scale=1.0)
    else:
        x_mat = data[[x]].to_numpy()
        kernel = DotProduct()
    y_vec = data[y].to_numpy()
    gp = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True)
    gp.fit(x_mat, y_vec)
    residual = y_vec - gp.predict(x_mat)
    if z:
        zmat = data[z].to_numpy()
        zmat = np.column_stack([np.ones(len(zmat)), zmat])
        bx, *_ = np.linalg.lstsq(zmat, data[x].to_numpy(), rcond=None)
        x_resid = data[x].to_numpy() - zmat @ bx
        by, *_ = np.linalg.lstsq(zmat, residual, rcond=None)
        y_resid = residual - zmat @ by
        _, p = pearsonr(x_resid, y_resid)
    else:
        _, p = pearsonr(data[x].to_numpy(), residual)
    return float(p)

