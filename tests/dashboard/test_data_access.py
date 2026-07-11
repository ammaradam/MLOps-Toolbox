from __future__ import annotations

import pytest

pytest.importorskip("mlflow")

from mlops_toolbox.adapters import adapt
from mlops_toolbox.dashboard.data_access import (
    build_metric_series,
    list_registered_models,
    list_runs,
)
from mlops_toolbox.registry.mlflow_backend import MLflowModelRegistry
from mlops_toolbox.tracking.mlflow_backend import MLflowTracker


def _tracking_uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _seed_runs(tracking_uri: str) -> None:
    tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="demo")
    tracker.start_run(run_name="run-1")
    tracker.log_params({"n_estimators": 50})
    tracker.log_metrics({"accuracy": 0.8})
    tracker.end_run()

    tracker.start_run(run_name="run-2")
    tracker.log_params({"n_estimators": 100})
    tracker.log_metrics({"accuracy": 0.9})
    tracker.end_run()


def test_list_runs_returns_newest_first(tmp_path) -> None:
    uri = _tracking_uri(tmp_path)
    _seed_runs(uri)

    runs = list_runs(uri)
    assert len(runs) == 2
    assert runs[0].run_name == "run-2"
    assert runs[1].run_name == "run-1"
    assert runs[0].metrics["accuracy"] == 0.9
    assert runs[0].status == "FINISHED"
    assert runs[0].start_time is not None


def test_list_runs_on_store_with_no_runs_returns_empty(tmp_path) -> None:
    uri = _tracking_uri(tmp_path)
    assert list_runs(uri) == []


def test_list_registered_models_resolves_latest_version(tmp_path, dummy_model) -> None:
    uri = _tracking_uri(tmp_path)
    registry = MLflowModelRegistry(tracking_uri=uri)
    adapter = adapt(dummy_model)
    registry.register_model(adapter, name="churn-model")
    registry.register_model(adapter, name="churn-model")  # bump to version 2

    models = list_registered_models(uri)
    assert len(models) == 1
    assert models[0].name == "churn-model"
    assert models[0].version == "2"
    assert models[0].backend == "mlflow"


def test_build_metric_series_is_chronological(tmp_path) -> None:
    uri = _tracking_uri(tmp_path)
    _seed_runs(uri)

    runs = list_runs(uri)  # newest-first
    series = build_metric_series(runs)

    assert "accuracy" in series
    values = [value for _, value in series["accuracy"]]
    assert values == [0.8, 0.9]  # oldest first, regardless of list_runs ordering
