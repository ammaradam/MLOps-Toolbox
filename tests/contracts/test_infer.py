from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

from mlops_toolbox.contracts import infer_contract, row_model_from_contract
from mlops_toolbox.core.models import EvaluationReport, ModelContract


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "count": np.array([1, 2], dtype=np.int64),
            "amount": np.array([1.5, 2.5], dtype=np.float64),
            "active": np.array([True, False]),
            "label": ["a", "b"],
        }
    )


def test_infer_contract_maps_dtypes() -> None:
    contract = infer_contract(_frame())
    assert [(c.name, c.dtype) for c in contract.input_columns] == [
        ("count", "integer"),
        ("amount", "float"),
        ("active", "boolean"),
        ("label", "string"),
    ]
    assert contract.task_type is None
    assert contract.evaluation is None


def test_infer_contract_carries_evaluation() -> None:
    report = EvaluationReport(task_type="classification", metrics={"accuracy": 0.9}, n_samples=10)
    contract = infer_contract(_frame(), evaluation=report)
    assert contract.task_type == "classification"
    assert contract.evaluation is not None
    assert contract.evaluation.metrics["accuracy"] == 0.9


def test_infer_contract_rejects_ndarray() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        infer_contract(np.zeros((2, 2)))  # type: ignore[arg-type]


def test_contract_json_round_trip() -> None:
    contract = infer_contract(_frame())
    restored = ModelContract.model_validate_json(contract.model_dump_json())
    assert restored == contract


def test_row_model_from_contract_validates_types() -> None:
    row_model = row_model_from_contract(infer_contract(_frame()))
    row = row_model(count=3, amount=0.5, active=True, label="a")
    assert row.model_dump() == {"count": 3, "amount": 0.5, "active": True, "label": "a"}
    with pytest.raises(pydantic.ValidationError):
        row_model(count="not-an-int", amount=0.5, active=True, label="a")
    with pytest.raises(pydantic.ValidationError):
        row_model(count=3, amount=0.5, active=True)  # missing field


def test_infer_contract_captures_categories_and_ranges() -> None:
    contract = infer_contract(_frame())
    by_name = {spec.name: spec for spec in contract.input_columns}
    assert by_name["label"].categories == ["a", "b"]
    assert by_name["count"].min_value == 1.0
    assert by_name["count"].max_value == 2.0
    assert by_name["label"].min_value is None
    assert contract.schema_version == 2


def test_infer_contract_skips_categories_for_high_cardinality() -> None:
    df = pd.DataFrame({"id": [f"user-{i}" for i in range(50)]})
    contract = infer_contract(df)
    assert contract.input_columns[0].categories is None


def test_infer_contract_output_spec_from_y() -> None:
    contract = infer_contract(_frame(), y=np.array([0, 1, 1, 0]))
    assert contract.output is not None
    assert contract.output.dtype == "integer"
    assert contract.output.labels == ["0", "1"]

    regression = infer_contract(_frame(), y=np.array([1.5, 2.5]))
    assert regression.output is not None
    assert regression.output.dtype == "float"
    assert regression.output.labels is None


def test_row_model_rejects_unknown_category() -> None:
    row_model = row_model_from_contract(infer_contract(_frame()))
    with pytest.raises(pydantic.ValidationError):
        row_model(count=3, amount=0.5, active=True, label="never-seen")
