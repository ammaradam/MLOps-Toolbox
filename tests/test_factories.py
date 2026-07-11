from __future__ import annotations

import pytest

import mlops_toolbox as mt


def test_factories_return_base_interfaces(tmp_path) -> None:
    pytest.importorskip("mlflow")
    pytest.importorskip("fastapi")
    pytest.importorskip("evidently")

    uri = f"sqlite:///{tmp_path / 'store.db'}"
    assert isinstance(mt.tracker(uri), mt.ExperimentTracker)
    assert isinstance(mt.model_registry(uri), mt.ModelRegistry)
    assert isinstance(mt.drift_detector(), mt.DriftDetector)
    assert isinstance(mt.model_server(), mt.ModelServer)
