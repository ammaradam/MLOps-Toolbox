from __future__ import annotations

from typing import Any

from mlops_toolbox.adapters.base import detect_framework
from mlops_toolbox.adapters.sklearn_adapter import SklearnAdapter
from mlops_toolbox.core.protocols import ModelAdapter

__all__ = ["adapt", "SklearnAdapter", "detect_framework"]


def adapt(model: Any) -> ModelAdapter:
    """Wrap an arbitrary trained model as a ModelAdapter.

    If `model` already satisfies the ModelAdapter protocol it is returned
    unchanged; otherwise it is wrapped in a duck-typed SklearnAdapter, which
    works for any object exposing predict()/predict_proba().
    """
    if isinstance(model, ModelAdapter):
        return model
    if not hasattr(model, "predict"):
        raise TypeError(f"{type(model)!r} has no predict() method and cannot be adapted.")
    return SklearnAdapter(model, framework=detect_framework(model))
