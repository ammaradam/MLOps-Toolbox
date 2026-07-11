from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import pandas as pd
from fastapi.testclient import TestClient

from mlops_toolbox.adapters import adapt
from mlops_toolbox.contracts import infer_contract
from mlops_toolbox.deployment.fastapi_backend import FastAPIModelServer


def test_health_endpoint(dummy_model) -> None:
    server = FastAPIModelServer()
    app = server.build_app(adapt(dummy_model))
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(dummy_model) -> None:
    server = FastAPIModelServer()
    app = server.build_app(adapt(dummy_model))
    client = TestClient(app)

    response = client.post("/predict", json={"instances": [[1.0], [-1.0]]})
    assert response.status_code == 200
    assert response.json() == {"predictions": [1, 0]}


def _contract_client(dummy_model) -> TestClient:
    X = pd.DataFrame({"feature_a": [1.0, -1.0], "feature_b": [3.0, 4.0]})
    contract = infer_contract(X)
    app = FastAPIModelServer().build_app(adapt(dummy_model), contract=contract)
    return TestClient(app)


def test_contract_predict_accepts_typed_records(dummy_model) -> None:
    client = _contract_client(dummy_model)
    records = [{"feature_a": 1.0, "feature_b": 3.0}, {"feature_a": -1.0, "feature_b": 4.0}]
    response = client.post("/predict", json={"records": records})
    assert response.status_code == 200
    assert response.json() == {"predictions": [1, 0]}


def test_contract_predict_rejects_bad_payloads(dummy_model) -> None:
    client = _contract_client(dummy_model)
    missing_field = client.post("/predict", json={"records": [{"feature_a": 1.0}]})
    assert missing_field.status_code == 422
    wrong_type = client.post(
        "/predict", json={"records": [{"feature_a": "not-a-float", "feature_b": 3.0}]}
    )
    assert wrong_type.status_code == 422


def test_contract_predict_rejects_unknown_category(dummy_model) -> None:
    X = pd.DataFrame({"feature_a": [1.0, -1.0], "segment": ["a", "b"]})
    app = FastAPIModelServer().build_app(adapt(dummy_model), contract=infer_contract(X))
    client = TestClient(app)

    ok = client.post("/predict", json={"records": [{"feature_a": 1.0, "segment": "a"}]})
    assert ok.status_code == 200
    unknown = client.post("/predict", json={"records": [{"feature_a": 1.0, "segment": "z"}]})
    assert unknown.status_code == 422


def test_contract_endpoint_returns_schema(dummy_model) -> None:
    client = _contract_client(dummy_model)
    response = client.get("/contract")
    assert response.status_code == 200
    payload = response.json()
    assert [col["name"] for col in payload["input_columns"]] == ["feature_a", "feature_b"]
    assert payload["schema_version"] == 2
