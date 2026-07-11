from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mlops_toolbox.core.models import ColumnSpec, EvaluationReport, ModelContract, OutputSpec

# String columns with at most this many distinct values are treated as a closed
# category set and enforced at serving/scoring time.
_MAX_CATEGORIES = 20


def _spec_dtype(dtype: object) -> Literal["integer", "float", "boolean", "string"]:
    kind = getattr(dtype, "kind", None)
    if kind in ("i", "u"):
        return "integer"
    if kind == "f":
        return "float"
    if kind == "b":
        return "boolean"
    return "string"


def _column_spec(column: pd.Series, name: str) -> ColumnSpec:
    dtype = _spec_dtype(column.dtype)
    categories: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None
    if dtype == "string":
        unique = column.dropna().unique()
        if 0 < len(unique) <= _MAX_CATEGORIES:
            categories = sorted(str(value) for value in unique)
    elif dtype in ("integer", "float") and len(column):
        min_value = float(column.min())
        max_value = float(column.max())
    return ColumnSpec(
        name=name, dtype=dtype, categories=categories, min_value=min_value, max_value=max_value
    )


def _output_spec(y: Any) -> OutputSpec:
    values = np.asarray(y)
    dtype = _spec_dtype(values.dtype)
    labels: list[str] | None = None
    if dtype != "float":
        unique = np.unique(values)
        if len(unique) <= _MAX_CATEGORIES:
            labels = [str(value) for value in unique]
    return OutputSpec(dtype=dtype, labels=labels)


def infer_contract(
    X: pd.DataFrame,
    y: Any | None = None,
    evaluation: EvaluationReport | None = None,
) -> ModelContract:
    """Infer a ModelContract from the training features (and optionally targets/eval).

    Column names and dtypes become the input schema serving validates against;
    low-cardinality string columns get a closed category set; numeric columns
    record their observed range. Pass `y` to also describe the model's output.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "infer_contract() requires a pandas DataFrame (column names carry the input "
            f"schema); got {type(X).__name__}. Wrap your array in a DataFrame with named "
            "columns first."
        )
    columns = [_column_spec(X[col], str(col)) for col in X.columns]
    return ModelContract(
        input_columns=columns,
        output=_output_spec(y) if y is not None else None,
        task_type=evaluation.task_type if evaluation is not None else None,
        evaluation=evaluation,
    )
