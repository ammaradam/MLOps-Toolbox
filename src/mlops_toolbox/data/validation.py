from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel, TypeAdapter, ValidationError

from mlops_toolbox.core.models import ValidationResult


class DataValidator(ABC):
    @abstractmethod
    def validate(self, df: pd.DataFrame, schema: Any) -> ValidationResult: ...


class PydanticDataFrameValidator(DataValidator):
    """Validates each row of a dataframe against a per-row pydantic schema.

    Validation runs as a single pydantic-core pass over the whole frame
    (TypeAdapter over list[schema]) rather than a Python-level loop per row.
    """

    def validate(self, df: pd.DataFrame, schema: type[BaseModel]) -> ValidationResult:
        adapter: TypeAdapter[list[BaseModel]] = TypeAdapter(list[schema])  # type: ignore[valid-type]
        errors: list[str] = []
        try:
            adapter.validate_python(df.to_dict(orient="records"))
        except ValidationError as exc:
            for err in exc.errors():
                row_idx, *field_loc = err["loc"]
                loc = ".".join(str(part) for part in field_loc)
                errors.append(f"row {row_idx} [{loc}]: {err['msg']}")
        return ValidationResult(is_valid=not errors, errors=errors, schema_name=schema.__name__)


def validate_dataframe(df: pd.DataFrame, schema: type[BaseModel]) -> ValidationResult:
    return PydanticDataFrameValidator().validate(df, schema)
