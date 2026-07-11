from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlops_toolbox.cli._app import app

runner = CliRunner()


def _ship(tracking_uri: str, name: str, out: Path, *extra: str) -> object:
    args = ["ship", name, "--tracking-uri", tracking_uri, "--out", str(out), *extra]
    return runner.invoke(app, args)


def test_ship_creates_self_contained_folder(registered_model_store, tmp_path) -> None:
    uri, name = registered_model_store
    out = tmp_path / "shipped"

    result = _ship(uri, name, out)
    assert result.exit_code == 0, result.output

    assert (out / "model" / "MLmodel").exists()
    assert (out / "contract.json").exists()
    for file_name in ("serve.py", "requirements.txt", "Dockerfile", "README.md"):
        assert (out / file_name).exists()
    requirements = (out / "requirements.txt").read_text(encoding="utf-8")
    assert "mlops-toolbox[deploy,registry]" in requirements


def test_ship_refuses_non_empty_out_without_force(registered_model_store, tmp_path) -> None:
    uri, name = registered_model_store
    out = tmp_path / "shipped"
    out.mkdir()
    (out / "existing.txt").write_text("keep me", encoding="utf-8")

    result = _ship(uri, name, out)
    assert result.exit_code == 1

    forced = _ship(uri, name, out, "--force")
    assert forced.exit_code == 0, forced.output


def test_shipped_serve_py_predicts(registered_model_store, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    uri, name = registered_model_store
    out = tmp_path / "shipped"
    result = _ship(uri, name, out)
    assert result.exit_code == 0, result.output

    spec = importlib.util.spec_from_file_location("generated_serve", out / "serve.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_serve"] = module
    try:
        spec.loader.exec_module(module)
        client = TestClient(module.app)
        records = [{"feature_a": 1.0, "feature_b": 3.0}, {"feature_a": -1.0, "feature_b": 4.0}]
        response = client.post("/predict", json={"records": records})
        assert response.status_code == 200
        assert response.json() == {"predictions": [1, 0]}
        assert client.get("/contract").status_code == 200
    finally:
        del sys.modules["generated_serve"]
