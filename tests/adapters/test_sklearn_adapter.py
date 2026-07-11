from __future__ import annotations

import numpy as np
import pytest

from mlops_toolbox.adapters import SklearnAdapter, adapt


def test_adapt_wraps_duck_typed_model(dummy_model) -> None:
    adapter = adapt(dummy_model)
    assert isinstance(adapter, SklearnAdapter)
    assert adapter.framework == "custom"

    X = np.array([[1.0], [-1.0]])
    preds = adapter.predict(X)
    assert preds.tolist() == [1, 0]

    proba = adapter.predict_proba(X)
    assert proba is not None
    assert proba.shape == (2, 2)


def test_adapt_returns_already_adapted_model_unchanged(dummy_model) -> None:
    adapter = adapt(dummy_model)
    assert adapt(adapter) is adapter


def test_adapt_rejects_object_without_predict() -> None:
    with pytest.raises(TypeError):
        adapt(object())


def test_predict_proba_returns_none_when_unavailable() -> None:
    class NoProbaModel:
        def predict(self, X):
            return np.zeros(len(X))

    adapter = adapt(NoProbaModel())
    assert adapter.predict_proba(np.zeros((3, 1))) is None
