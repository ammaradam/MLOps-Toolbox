from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")
pytest.importorskip("evidently")

from mlops_toolbox.adapters import adapt
from mlops_toolbox.core.exceptions import MonitoringError
from mlops_toolbox.monitoring import check_drift, check_prediction_drift
from mlops_toolbox.registry.mlflow_backend import MLflowModelRegistry


def test_check_drift_uses_stored_reference(
    tmp_path, dummy_model, sample_dataframe_pair_with_drift
) -> None:
    reference, current = sample_dataframe_pair_with_drift
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MLflowModelRegistry(tracking_uri=uri)
    registry.register_model(adapt(dummy_model), name="drift-model", reference_data=reference)

    report = check_drift(current, registry=registry, name="drift-model")
    assert report.dataset_drift_detected is True
    assert report.column_results["drifted_feature"].drift_detected is True


def test_check_drift_without_reference_raises(tmp_path, dummy_model) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MLflowModelRegistry(tracking_uri=uri)
    registry.register_model(adapt(dummy_model), name="no-ref-model")

    with pytest.raises(MonitoringError, match="no stored reference"):
        check_drift(pd.DataFrame({"a": [1.0, 2.0]}), registry=registry, name="no-ref-model")
    with pytest.raises(MonitoringError, match="no stored reference predictions"):
        check_prediction_drift([0, 1], registry=registry, name="no-ref-model")


def test_check_prediction_drift_uses_stored_baseline(
    tmp_path, dummy_model, sample_dataframe_pair_with_drift
) -> None:
    reference, _ = sample_dataframe_pair_with_drift
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MLflowModelRegistry(tracking_uri=uri)
    adapter = adapt(dummy_model)
    registry.register_model(adapter, name="pred-drift-model", reference_data=reference)

    stored = registry.get_reference_predictions("pred-drift-model")
    assert stored is not None
    assert len(stored) == len(reference)

    # Same distribution of predictions -> no drift.
    same = check_prediction_drift(
        adapter.predict(reference), registry=registry, name="pred-drift-model"
    )
    assert same.dataset_drift_detected is False

    # All-ones predictions against a mixed baseline -> drift.
    drifted = check_prediction_drift(
        np.ones(len(reference), dtype=int), registry=registry, name="pred-drift-model"
    )
    assert drifted.dataset_drift_detected is True
