"""Batch scoring with contract enforcement — the non-endpoint path to predictions."""

from __future__ import annotations

import pandas as pd

from mlops_toolbox.contracts.validate import validate_against_contract
from mlops_toolbox.core.exceptions import ValidationFailedError
from mlops_toolbox.registry.base import ModelRegistry

__all__ = ["score_dataframe"]


def score_dataframe(
    df: pd.DataFrame,
    *,
    registry: ModelRegistry,
    name: str,
    version: str = "latest",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Score a dataframe with a registered model, enforcing its contract on the way in.

    Inputs are validated against the stored contract (columns present, types
    compatible, categories known) and passed to the model in contract column
    order; extra columns are carried through untouched. Returns a copy of `df`
    with the predictions appended. Raises ValidationFailedError listing every
    problem when the input violates the contract.
    """
    contract = registry.get_contract(name, version)
    model = registry.get_model(name, version)

    if contract is not None:
        problems = validate_against_contract(df, contract)
        if problems:
            raise ValidationFailedError(
                f"Input violates the contract of model '{name}' (version={version!r}):\n  - "
                + "\n  - ".join(problems)
            )
        X: pd.DataFrame = df[[spec.name for spec in contract.input_columns]]
    else:
        X = df

    scored = df.copy()
    scored[prediction_column] = model.predict(X)
    return scored
