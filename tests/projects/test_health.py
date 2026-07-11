from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from mlops_toolbox.adapters import adapt
from mlops_toolbox.contracts import infer_contract
from mlops_toolbox.core.models import EvaluationReport
from mlops_toolbox.projects.health import audit_project
from mlops_toolbox.registry.mlflow_backend import MLflowModelRegistry
from mlops_toolbox.tracking.mlflow_backend import MLflowTracker


def _status_by_name(health) -> dict[str, str]:
    return {check.name: check.status for check in health.checks}


def test_audit_unreachable_store_fails_first_check() -> None:
    # A local sqlite URI is auto-created by MLflow on connection, so an invalid
    # URI is the realistic "unreachable" case.
    health = audit_project("not-a-valid-tracking-uri")
    assert health.passed is False
    assert health.checks[0].name == "tracking store reachable"
    assert health.checks[0].status == "fail"


def test_audit_empty_store(tmp_path) -> None:
    # Instantiating a tracker creates the store but leaves it without runs or models.
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    MLflowTracker(tracking_uri=uri, experiment_name="empty")

    health = audit_project(uri)
    statuses = _status_by_name(health)
    assert statuses["tracking store reachable"] == "pass"
    assert statuses["experiment runs recorded"] == "fail"
    assert statuses["model registered"] == "fail"
    assert health.passed is False
    failing = [check for check in health.checks if check.status == "fail"]
    assert all(check.hint for check in failing)


def test_audit_model_without_contract(tmp_path, dummy_model) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MLflowTracker(tracking_uri=uri, experiment_name="demo")
    with tracker:
        tracker.log_metrics({"accuracy": 0.9})
    MLflowModelRegistry(tracking_uri=uri).register_model(adapt(dummy_model), name="bare")

    health = audit_project(uri)
    statuses = _status_by_name(health)
    assert statuses["experiment runs recorded"] == "pass"
    assert statuses["runs log metrics"] == "pass"
    assert statuses["model registered"] == "pass"
    assert statuses["model carries a contract"] == "fail"
    assert statuses["drift reference stored"] == "fail"
    assert health.passed is False


def test_audit_fully_contracted_model_passes(tmp_path, dummy_model) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MLflowTracker(tracking_uri=uri, experiment_name="demo")
    with tracker:
        tracker.log_metrics({"accuracy": 0.9})

    X = pd.DataFrame({"feature_a": [1.0, -1.0]})
    report = EvaluationReport(task_type="classification", metrics={"accuracy": 0.9}, n_samples=2)
    MLflowModelRegistry(tracking_uri=uri).register_model(
        adapt(dummy_model),
        name="full",
        contract=infer_contract(X, evaluation=report),
        reference_data=X,
    )

    health = audit_project(uri)
    assert health.passed is True
    assert health.n_fail == 0
    assert health.n_warn == 0
    assert _status_by_name(health)["contract records evaluation"] == "pass"
