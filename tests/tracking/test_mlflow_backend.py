from __future__ import annotations

import pytest

pytest.importorskip("mlflow")

from mlops_toolbox.core.exceptions import TrackingError
from mlops_toolbox.tracking.mlflow_backend import MLflowTracker


def _tracker(tmp_path) -> MLflowTracker:
    return MLflowTracker(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", experiment_name="test")


def test_run_lifecycle_logs_params_and_metrics(tmp_path) -> None:
    tracker = _tracker(tmp_path)
    run_id = tracker.start_run(run_name="unit-test")
    assert run_id

    tracker.log_params({"n_estimators": 100})
    tracker.log_metrics({"accuracy": 0.9})
    result = tracker.end_run()

    assert result.run_id == run_id
    assert result.backend == "mlflow"
    assert result.status == "FINISHED"
    assert result.params == {"n_estimators": 100}
    assert result.metrics == {"accuracy": 0.9}
    assert result.end_time is not None


def test_context_manager_marks_failed_on_exception(tmp_path) -> None:
    tracker = _tracker(tmp_path)
    with pytest.raises(RuntimeError), tracker:
        raise RuntimeError("boom")


def test_logging_without_active_run_raises(tmp_path) -> None:
    tracker = _tracker(tmp_path)
    with pytest.raises(TrackingError):
        tracker.log_params({"x": 1})
