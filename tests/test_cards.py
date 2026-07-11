from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from mlops_toolbox.adapters import adapt
from mlops_toolbox.cards import generate_model_card
from mlops_toolbox.contracts import infer_contract
from mlops_toolbox.core.models import EvaluationReport
from mlops_toolbox.factories import model_registry


def test_model_card_contains_schema_metrics_and_monitoring(tmp_path, dummy_model) -> None:
    registry = model_registry(f"sqlite:///{tmp_path / 'store.db'}")
    X = pd.DataFrame({"feature_a": [1.0, -1.0], "segment": ["a", "b"]})
    report = EvaluationReport(task_type="classification", metrics={"accuracy": 0.9}, n_samples=2)
    info = registry.register_model(
        adapt(dummy_model),
        name="card-model",
        contract=infer_contract(X, y=[0, 1], evaluation=report),
        reference_data=X,
        tags={"team": "growth"},
    )
    registry.set_alias("card-model", info.version, "production")

    card = generate_model_card(registry, "card-model")
    assert "# Model card: card-model" in card
    assert "production" in card
    assert "`feature_a` | float" in card
    assert "one of: `a`, `b`" in card
    assert "| accuracy | 0.9 |" in card
    assert "Drift baseline stored" in card
    assert "team=growth" in card
    assert "**Output**: integer" in card


def test_model_card_without_contract_names_the_gap(tmp_path, dummy_model) -> None:
    registry = model_registry(f"sqlite:///{tmp_path / 'store.db'}")
    registry.register_model(adapt(dummy_model), name="bare-model")

    card = generate_model_card(registry, "bare-model")
    assert "No contract stored" in card
    assert "No drift baseline stored" in card
