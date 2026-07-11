from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

from mlops_toolbox.core.models import DatasetSplit


def train_test_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> DatasetSplit:
    """Shuffle-split X/y into train/test sets. Native numpy implementation, no sklearn."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(f"X and y must have the same length, got {n_samples} and {len(y)}")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    n_test = int(round(n_samples * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    return DatasetSplit(
        X_train=_select(X, train_idx),
        X_test=_select(X, test_idx),
        y_train=_select(y, train_idx),
        y_test=_select(y, test_idx),
    )


@overload
def _select(data: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame: ...
@overload
def _select(data: pd.Series, idx: np.ndarray) -> pd.Series: ...
@overload
def _select(data: np.ndarray, idx: np.ndarray) -> np.ndarray: ...
def _select(
    data: pd.DataFrame | pd.Series | np.ndarray, idx: np.ndarray
) -> pd.DataFrame | pd.Series | np.ndarray:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[idx].reset_index(drop=True)
    result: np.ndarray = np.asarray(data)[idx]
    return result
