from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mlops_toolbox.cli._app import app

runner = CliRunner()


@pytest.fixture
def tracked_run(tmp_path, dummy_model) -> tuple[str, str]:
    """A tracking store with one finished run holding metrics. Returns (uri, run_id)."""
    from mlops_toolbox.tracking.mlflow_backend import MLflowTracker

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = MLflowTracker(tracking_uri=uri, experiment_name="gate-test")
    run_id = tracker.start_run()
    tracker.log_metrics({"accuracy": 0.92, "f1_macro": 0.80})
    tracker.end_run()
    return uri, run_id


def test_gate_passes_and_exits_zero(tracked_run) -> None:
    uri, run_id = tracked_run
    result = runner.invoke(
        app, ["gate", "accuracy>=0.9", "--run", run_id, "--tracking-uri", uri]
    )
    assert result.exit_code == 0, result.output
    assert "[PASS]" in result.output


def test_gate_fails_and_exits_one(tracked_run) -> None:
    uri, run_id = tracked_run
    result = runner.invoke(
        app,
        ["gate", "accuracy>=0.9", "f1_macro>=0.85", "--run", run_id, "--tracking-uri", uri],
    )
    assert result.exit_code == 1
    assert "[FAIL]" in result.output


def test_gate_defaults_to_latest_finished_run(tracked_run) -> None:
    uri, _ = tracked_run
    result = runner.invoke(app, ["gate", "accuracy>0.5", "--tracking-uri", uri])
    assert result.exit_code == 0, result.output
    assert "latest finished run" in result.output


def test_gate_resolves_model_source_run(registered_model_store, dummy_model) -> None:
    uri, name = registered_model_store
    # The registry's own run holds no metrics, so the gate on a real metric must fail
    # with the "not found" message rather than crash.
    result = runner.invoke(
        app, ["gate", "accuracy>=0.5", "--model", name, "--tracking-uri", uri]
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_gate_rejects_run_and_model_together(tracked_run) -> None:
    uri, run_id = tracked_run
    result = runner.invoke(
        app, ["gate", "accuracy>=0.5", "--run", run_id, "--model", "x", "--tracking-uri", uri]
    )
    assert result.exit_code != 0
    assert "not both" in result.output
