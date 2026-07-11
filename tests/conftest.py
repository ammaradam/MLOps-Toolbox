from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


class DummyModel:
    """Hand-rolled model exposing predict()/predict_proba(), no framework dependency."""

    def __init__(self, n_classes: int = 2) -> None:
        self.n_classes = n_classes

    def predict(self, X: Any) -> np.ndarray:
        X = np.asarray(X)
        return (X[:, 0] > 0).astype(int) if self.n_classes == 2 else np.zeros(len(X), dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X)
        p1 = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p1, p1])


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel()


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        {
            "feature_a": rng.normal(size=n),
            "feature_b": rng.normal(size=n),
        }
    )
    y = (X["feature_a"] + 0.5 * X["feature_b"] > 0).astype(int).to_numpy()
    return X, y


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"feature_a": rng.normal(size=n)})
    noise = rng.normal(scale=0.1, size=n)
    y = (3.0 * X["feature_a"] + 1.0 + noise).to_numpy()
    return X, y


@pytest.fixture
def sample_dataframe_pair_with_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 300
    reference = pd.DataFrame(
        {
            "stable_feature": rng.normal(loc=0.0, scale=1.0, size=n),
            "drifted_feature": rng.normal(loc=0.0, scale=1.0, size=n),
        }
    )
    current = pd.DataFrame(
        {
            "stable_feature": rng.normal(loc=0.0, scale=1.0, size=n),
            "drifted_feature": rng.normal(loc=10.0, scale=1.0, size=n),
        }
    )
    return reference, current
