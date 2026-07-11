from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SklearnAdapter:
    """Duck-typed ModelAdapter for any object exposing predict()/predict_proba().

    Despite the name, this never imports scikit-learn — it works for any
    estimator that follows the fit/predict convention (scikit-learn, XGBoost's
    and LightGBM's sklearn APIs, and hand-rolled models alike).
    """

    def __init__(self, model: Any, framework: str = "custom") -> None:
        self._model = model
        self.framework = framework

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        proba_fn = getattr(self._model, "predict_proba", None)
        if proba_fn is None:
            return None
        return np.asarray(proba_fn(X))

    @property
    def raw(self) -> Any:
        return self._model
