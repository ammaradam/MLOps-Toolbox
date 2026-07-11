from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from mlops_toolbox.adapters import adapt
from mlops_toolbox.contracts import infer_contract
from mlops_toolbox.core.exceptions import ValidationFailedError
from mlops_toolbox.factories import model_registry
from mlops_toolbox.scoring import score_dataframe


@pytest.fixture
def scoring_registry(tmp_path, dummy_model):
    registry = model_registry(f"sqlite:///{tmp_path / 'store.db'}")
    X = pd.DataFrame({"feature_a": [1.0, -1.0, 2.0], "feature_b": [3.0, 4.0, 5.0]})
    registry.register_model(
        adapt(dummy_model), name="scorer", contract=infer_contract(X), reference_data=X
    )
    return registry


def test_score_dataframe_appends_predictions(scoring_registry) -> None:
    df = pd.DataFrame(
        {
            "feature_b": [9.0, 9.0],  # wrong order on purpose: contract fixes column order
            "feature_a": [1.0, -1.0],
            "customer_id": ["c1", "c2"],  # extra column carried through
        }
    )
    scored = score_dataframe(df, registry=scoring_registry, name="scorer")
    assert scored["prediction"].tolist() == [1, 0]
    assert scored["customer_id"].tolist() == ["c1", "c2"]
    assert len(df.columns) == 3  # input untouched


def test_score_dataframe_rejects_contract_violations(scoring_registry) -> None:
    df = pd.DataFrame({"feature_a": [1.0]})  # feature_b missing
    with pytest.raises(ValidationFailedError, match="missing column 'feature_b'"):
        score_dataframe(df, registry=scoring_registry, name="scorer")


def test_score_dataframe_custom_prediction_column(scoring_registry) -> None:
    df = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]})
    scored = score_dataframe(
        df, registry=scoring_registry, name="scorer", prediction_column="churn_pred"
    )
    assert "churn_pred" in scored.columns
