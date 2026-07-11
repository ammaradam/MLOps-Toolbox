from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ModelAdapter(Protocol):
    """Structural interface every framework-specific model wrapper satisfies.

    Downstream stages (evaluation, monitoring, deployment) depend only on this
    protocol, never on a specific ML framework's model class.
    """

    framework: str

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None: ...
