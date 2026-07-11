from __future__ import annotations

import pandas as pd

from mlops_toolbox.contracts.infer import _spec_dtype
from mlops_toolbox.core.models import ColumnSpec, ModelContract

# Which actual dtypes satisfy each spec dtype (integers are acceptable floats).
_COMPATIBLE: dict[str, set[str]] = {
    "integer": {"integer"},
    "float": {"float", "integer"},
    "boolean": {"boolean"},
    "string": {"string"},
}


def validate_against_contract(df: pd.DataFrame, contract: ModelContract) -> list[str]:
    """Check a dataframe against a contract's input schema; returns problems (empty = valid).

    Validates presence, dtype compatibility, and closed category sets. Numeric
    ranges are informational and deliberately not enforced — unseen-but-valid
    values must not be rejected at scoring time.
    """
    problems: list[str] = []
    for spec in contract.input_columns:
        if spec.name not in df.columns:
            problems.append(f"missing column '{spec.name}' ({spec.dtype})")
            continue
        column = df[spec.name]
        actual = _spec_dtype(column.dtype)
        if actual not in _COMPATIBLE[spec.dtype]:
            problems.append(f"column '{spec.name}': expected {spec.dtype}, got {actual}")
            continue
        problems.extend(_category_problems(column, spec))
    return problems


def _category_problems(column: pd.Series, spec: ColumnSpec) -> list[str]:
    if not spec.categories:
        return []
    allowed = set(spec.categories)
    unknown = sorted({str(v) for v in column.dropna().unique()} - allowed)
    if not unknown:
        return []
    shown = ", ".join(unknown[:5]) + (", ..." if len(unknown) > 5 else "")
    return [f"column '{spec.name}': unknown categories [{shown}]; allowed: {sorted(allowed)}"]
