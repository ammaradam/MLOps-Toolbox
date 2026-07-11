from __future__ import annotations

import pandas as pd
import pydantic

from mlops_toolbox.data import validate_dataframe


class RowSchema(pydantic.BaseModel):
    age: int = pydantic.Field(ge=0)
    name: str


def test_validate_dataframe_all_valid() -> None:
    df = pd.DataFrame({"age": [25, 30], "name": ["a", "b"]})
    result = validate_dataframe(df, RowSchema)
    assert result.is_valid is True
    assert result.errors == []
    assert result.schema_name == "RowSchema"


def test_validate_dataframe_reports_row_errors() -> None:
    df = pd.DataFrame({"age": [25, -5], "name": ["a", "b"]})
    result = validate_dataframe(df, RowSchema)
    assert result.is_valid is False
    assert len(result.errors) == 1
    assert "row 1" in result.errors[0]
