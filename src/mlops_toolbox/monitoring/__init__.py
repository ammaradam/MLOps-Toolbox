from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlops_toolbox.core.exceptions import MonitoringError
from mlops_toolbox.core.models import DriftReport
from mlops_toolbox.monitoring.base import DriftDetector
from mlops_toolbox.registry.base import ModelRegistry

__all__ = ["DriftDetector", "check_drift", "check_prediction_drift"]


def check_drift(
    current: pd.DataFrame,
    *,
    registry: ModelRegistry,
    name: str,
    version: str = "latest",
) -> DriftReport:
    """Compare current data against the reference dataset stored with a registered model.

    The reference is the `reference_data` passed to `register_model()` — the model
    carries its own drift baseline, so callers don't have to keep the training
    data around.
    """
    reference = registry.get_reference_data(name, version)
    if reference is None:
        raise MonitoringError(
            f"Model '{name}' (version={version!r}) has no stored reference dataset. "
            "Pass reference_data= when calling register_model() to enable drift checks."
        )
    from mlops_toolbox.factories import drift_detector

    return drift_detector().compare(reference, current)


def check_prediction_drift(
    current_predictions: Any,
    *,
    registry: ModelRegistry,
    name: str,
    version: str = "latest",
) -> DriftReport:
    """Compare recent predictions against the baseline predictions stored at registration.

    When `register_model()` receives `reference_data`, it also stores the model's
    predictions over that reference — so prediction drift can be checked later
    without re-running the original model.
    """
    reference = registry.get_reference_predictions(name, version)
    if reference is None:
        raise MonitoringError(
            f"Model '{name}' (version={version!r}) has no stored reference predictions. "
            "Pass reference_data= when calling register_model() to enable prediction-drift "
            "checks."
        )
    prediction_column = reference.columns[0]
    current = pd.DataFrame({prediction_column: np.asarray(current_predictions).ravel()})
    from mlops_toolbox.factories import drift_detector

    return drift_detector().compare(reference, current)
