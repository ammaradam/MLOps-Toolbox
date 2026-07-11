from __future__ import annotations

import pandas as pd

from mlops_toolbox.contracts import infer_contract, validate_against_contract


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [1.5, 2.5, 3.0],
            "count": [1, 2, 3],
            "segment": ["a", "b", "a"],
        }
    )


def test_valid_frame_has_no_problems() -> None:
    contract = infer_contract(_training_frame())
    assert validate_against_contract(_training_frame(), contract) == []


def test_integer_column_satisfies_float_spec() -> None:
    contract = infer_contract(_training_frame())
    df = _training_frame()
    df["amount"] = [1, 2, 3]  # ints where floats were seen in training
    assert validate_against_contract(df, contract) == []


def test_missing_column_and_wrong_type_reported() -> None:
    contract = infer_contract(_training_frame())
    df = pd.DataFrame({"amount": ["not", "a", "number"], "segment": ["a", "b", "a"]})
    problems = validate_against_contract(df, contract)
    assert any("missing column 'count'" in p for p in problems)
    assert any("'amount': expected float, got string" in p for p in problems)


def test_unknown_category_reported() -> None:
    contract = infer_contract(_training_frame())
    df = _training_frame()
    df.loc[0, "segment"] = "z"
    problems = validate_against_contract(df, contract)
    assert len(problems) == 1
    assert "unknown categories [z]" in problems[0]


def test_out_of_range_numeric_is_not_an_error() -> None:
    contract = infer_contract(_training_frame())
    df = _training_frame()
    df.loc[0, "amount"] = 1_000_000.0
    assert validate_against_contract(df, contract) == []