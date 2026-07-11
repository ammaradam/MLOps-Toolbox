from __future__ import annotations

from typing import Literal

import pydantic

from mlops_toolbox.core.models import ModelContract

_DTYPE_TO_PYTHON: dict[str, type] = {
    "integer": int,
    "float": float,
    "boolean": bool,
    "string": str,
}


def row_model_from_contract(contract: ModelContract) -> type[pydantic.BaseModel]:
    """Build a pydantic model for one input row from a contract's column specs.

    Serving uses this to give /predict a typed, self-documenting request body.
    Columns with a closed category set validate as literals, so unknown
    categories are rejected at the boundary.
    """
    fields: dict[str, tuple[object, object]] = {}
    for spec in contract.input_columns:
        annotation: object = _DTYPE_TO_PYTHON[spec.dtype]
        if spec.categories:
            annotation = Literal[tuple(spec.categories)]  # noqa: F821
        fields[spec.name] = (annotation, ...)
    return pydantic.create_model("PredictRecord", **fields)  # type: ignore[call-overload, no-any-return]
