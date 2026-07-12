from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mlflow")

from fastapi.testclient import TestClient

from mlops_toolbox.adapters import adapt
from mlops_toolbox.dashboard.app import create_app
from mlops_toolbox.projects import register_project
from mlops_toolbox.registry.mlflow_backend import MLflowModelRegistry
from mlops_toolbox.tracking.mlflow_backend import MLflowTracker


def _seed_project(tmp_path, dummy_model) -> tuple[str, str]:
    mlflow_dir = tmp_path / "mlflow-store"
    tracking_uri = f"sqlite:///{mlflow_dir / 'mlflow.db'}"

    tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="demo")
    tracker.start_run(run_name="seed-run-1")
    tracker.log_params({"n_estimators": 10})
    tracker.log_metrics({"accuracy": 0.80})
    tracker.end_run()

    tracker.start_run(run_name="seed-run")
    tracker.log_params({"n_estimators": 20})
    tracker.log_metrics({"accuracy": 0.87})
    run_result = tracker.end_run()

    registry = MLflowModelRegistry(tracking_uri=tracking_uri)
    registry.register_model(adapt(dummy_model), name="seed-model", run_id=run_result.run_id)

    registry_path = tmp_path / "projects.json"
    register_project("seed-project", tracking_uri, registry_path=registry_path)
    return str(registry_path), run_result.run_id


def test_health_endpoint(tmp_path, dummy_model) -> None:
    registry_path, _ = _seed_project(tmp_path, dummy_model)
    app = create_app(registry_path=registry_path)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_lists_project(tmp_path, dummy_model) -> None:
    registry_path, _ = _seed_project(tmp_path, dummy_model)
    app = create_app(registry_path=registry_path)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "seed-project" in response.text
    # Health badge: model registered without a contract -> at least one failing check.
    assert "fail" in response.text


def test_project_detail_shows_runs_and_models(tmp_path, dummy_model) -> None:
    registry_path, run_id = _seed_project(tmp_path, dummy_model)
    app = create_app(registry_path=registry_path)
    client = TestClient(app)

    response = client.get("/projects/seed-project")
    assert response.status_code == 200
    assert "seed-run" in response.text
    assert "seed-model" in response.text
    assert "accuracy" in response.text
    assert "<svg" in response.text
    assert run_id[:8] in response.text
    # Lifecycle health checklist renders with hints for the contract gap.
    assert "Lifecycle Health" in response.text
    assert "model carries a contract" in response.text
    assert "infer_contract" in response.text


def test_unknown_project_returns_404(tmp_path, dummy_model) -> None:
    registry_path, _ = _seed_project(tmp_path, dummy_model)
    app = create_app(registry_path=registry_path)
    client = TestClient(app)

    response = client.get("/projects/does-not-exist")
    assert response.status_code == 404


def test_project_edit_route_renames_project(tmp_path, dummy_model) -> None:
    registry_path, _ = _seed_project(tmp_path, dummy_model)
    app = create_app(registry_path=registry_path)
    client = TestClient(app)

    response = client.post(
        "/projects/seed-project/edit",
        data={"new_name": "renamed-project", "description": "now with description"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "/projects/renamed-project" in response.headers["location"]

    detail = client.get("/projects/renamed-project")
    assert detail.status_code == 200
    assert "now with description" in detail.text
    assert client.get("/projects/seed-project").status_code == 404


def test_project_edit_rejects_name_collision(tmp_path, dummy_model) -> None:
    registry_path, _ = _seed_project(tmp_path, dummy_model)
    register_project("other", "sqlite:///other.db", registry_path=registry_path)
    app = create_app(registry_path=registry_path)
    client = TestClient(app)

    response = client.post(
        "/projects/seed-project/edit",
        data={"new_name": "other", "description": ""},
        follow_redirects=False,
    )
    assert response.status_code == 409


def test_project_delete_route_unregisters(tmp_path, dummy_model) -> None:
    registry_path, _ = _seed_project(tmp_path, dummy_model)
    app = create_app(registry_path=registry_path)
    client = TestClient(app)

    response = client.post("/projects/seed-project/delete", follow_redirects=False)
    assert response.status_code == 303

    index = client.get("/")
    assert "seed-project" not in index.text
    assert client.post("/projects/seed-project/delete", follow_redirects=False).status_code == 404


def test_index_empty_state(tmp_path) -> None:
    registry_path = tmp_path / "empty-projects.json"
    app = create_app(registry_path=registry_path)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "No projects registered yet" in response.text
