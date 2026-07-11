from __future__ import annotations

from fastapi.testclient import TestClient

from mlops_toolbox.cli.serve import build_serving_app


def test_build_serving_app_uses_stored_contract(registered_model_store) -> None:
    uri, name = registered_model_store
    app, has_contract = build_serving_app(name, "latest", uri)
    assert has_contract is True

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    records = [{"feature_a": 1.0, "feature_b": 3.0}]
    response = client.post("/predict", json={"records": records})
    assert response.status_code == 200
    assert response.json() == {"predictions": [1]}
    assert client.post("/predict", json={"records": [{"feature_a": 1.0}]}).status_code == 422
