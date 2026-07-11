from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")

from mlops_toolbox.adapters import adapt
from mlops_toolbox.contracts import infer_contract
from mlops_toolbox.registry.mlflow_backend import MLflowModelRegistry
from mlops_toolbox.tracking.mlflow_backend import MLflowTracker


def _tracking_uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_register_and_load_model_round_trip(tmp_path, dummy_model) -> None:
    uri = _tracking_uri(tmp_path)
    tracker = MLflowTracker(tracking_uri=uri, experiment_name="registry-test")
    registry = MLflowModelRegistry(tracking_uri=uri)

    with tracker:
        tracker.log_params({"n_estimators": 10})
        info = registry.register_model(adapt(dummy_model), name="dummy-model")

    assert info.name == "dummy-model"
    assert info.version == "1"
    assert info.backend == "mlflow"

    loaded = registry.get_model("dummy-model", version="latest")
    X = np.array([[1.0], [-1.0]])
    preds = loaded.predict(X)
    assert preds.tolist() == [1, 0]


def test_register_with_contract_and_reference_round_trip(tmp_path, dummy_model) -> None:
    uri = _tracking_uri(tmp_path)
    registry = MLflowModelRegistry(tracking_uri=uri)
    X = pd.DataFrame({"feature_a": [1.0, -1.0], "feature_b": [3, 4]})
    contract = infer_contract(X)

    info = registry.register_model(
        adapt(dummy_model), name="contract-model", contract=contract, reference_data=X
    )
    assert info.tags["mlops_toolbox.has_contract"] == "true"
    assert info.tags["mlops_toolbox.has_reference"] == "true"

    restored = registry.get_contract("contract-model")
    assert restored is not None
    assert [c.name for c in restored.input_columns] == ["feature_a", "feature_b"]

    reference = registry.get_reference_data("contract-model")
    assert reference is not None
    pd.testing.assert_frame_equal(reference, X)


def test_contract_absent_returns_none(tmp_path, dummy_model) -> None:
    uri = _tracking_uri(tmp_path)
    registry = MLflowModelRegistry(tracking_uri=uri)

    registry.register_model(adapt(dummy_model), name="bare-model")
    assert registry.get_contract("bare-model") is None
    assert registry.get_reference_data("bare-model") is None


def test_reference_data_is_sampled_to_max_rows(tmp_path, dummy_model) -> None:
    uri = _tracking_uri(tmp_path)
    registry = MLflowModelRegistry(tracking_uri=uri)
    X = pd.DataFrame({"feature_a": np.arange(50, dtype=np.float64)})

    registry.register_model(
        adapt(dummy_model), name="sampled-model", reference_data=X, max_reference_rows=10
    )
    reference = registry.get_reference_data("sampled-model")
    assert reference is not None
    assert len(reference) == 10


def test_set_alias_and_list_versions(tmp_path, dummy_model) -> None:
    uri = _tracking_uri(tmp_path)
    registry = MLflowModelRegistry(tracking_uri=uri)
    adapter = adapt(dummy_model)

    info = registry.register_model(adapter, name="aliased-model")
    updated = registry.set_alias("aliased-model", info.version, "production")
    assert updated.aliases == ["production"]

    # Aliases resolve wherever a version is accepted.
    by_alias = registry.get_model_info("aliased-model", version="production")
    assert by_alias.version == info.version
    loaded = registry.get_model("aliased-model", version="production")
    assert loaded.predict(np.array([[1.0], [-1.0]])).tolist() == [1, 0]

    versions = registry.list_versions("aliased-model")
    assert len(versions) == 1
    assert versions[0].version == info.version
