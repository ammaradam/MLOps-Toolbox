from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("typer")
pytest.importorskip("mlflow")
pytest.importorskip("fastapi")

from mlops_toolbox.adapters import adapt
from mlops_toolbox.contracts import infer_contract
from mlops_toolbox.registry.mlflow_backend import MLflowModelRegistry


@pytest.fixture
def registered_model_store(tmp_path, dummy_model) -> tuple[str, str]:
    """A SQLite tracking store holding one contract-registered model.

    Returns (tracking_uri, model_name).
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MLflowModelRegistry(tracking_uri=uri)
    X = pd.DataFrame({"feature_a": [1.0, -1.0], "feature_b": [3.0, 4.0]})
    registry.register_model(
        adapt(dummy_model),
        name="cli-model",
        contract=infer_contract(X),
        reference_data=X,
    )
    return uri, "cli-model"
