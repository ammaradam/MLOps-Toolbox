from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class RunResult(BaseModel):
    run_id: str
    backend: str
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts_uri: str | None = None
    status: Literal["RUNNING", "FINISHED", "FAILED"] = "RUNNING"
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None


class ModelInfo(BaseModel):
    name: str
    version: str
    backend: str
    aliases: list[str] = Field(default_factory=list)
    source_run_id: str | None = None
    uri: str
    tags: dict[str, str] = Field(default_factory=dict)


class GateCheck(BaseModel):
    expression: str
    metric: str
    passed: bool
    actual: float | None = None
    message: str


class GateResult(BaseModel):
    passed: bool
    checks: list[GateCheck] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    task_type: Literal["classification", "regression"]
    metrics: dict[str, float] = Field(default_factory=dict)
    n_samples: int
    confusion_matrix: list[list[int]] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def check_gates(self, gates: Sequence[str]) -> GateResult:
        """Check threshold expressions like "f1_macro>=0.85" against this report's metrics."""
        from mlops_toolbox.evaluation.gates import check_gates

        return check_gates(self.metrics, gates)


class ColumnDriftResult(BaseModel):
    column: str
    drift_detected: bool
    drift_score: float
    stattest: str


class DriftReport(BaseModel):
    dataset_drift_detected: bool
    drift_share: float
    column_results: dict[str, ColumnDriftResult] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    schema_name: str


class ColumnSpec(BaseModel):
    name: str
    dtype: Literal["integer", "float", "boolean", "string"]
    # v2 value-level metadata, inferred from the training data when available.
    categories: list[str] | None = None  # closed value set; enforced at serving/scoring time
    min_value: float | None = None  # observed numeric range; informational, not enforced
    max_value: float | None = None


class OutputSpec(BaseModel):
    dtype: Literal["integer", "float", "boolean", "string"]
    labels: list[str] | None = None  # class labels for classification outputs


class ModelContract(BaseModel):
    """Everything a model needs downstream, captured once at registration time.

    Serving derives its request schema from `input_columns` (including closed
    category sets); monitoring pairs it with the reference dataset stored
    alongside; `evaluation` records the quality the model shipped with;
    `output` describes what predictions look like.
    """

    schema_version: int = 2
    input_columns: list[ColumnSpec]
    output: OutputSpec | None = None
    task_type: Literal["classification", "regression"] | None = None
    evaluation: EvaluationReport | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProjectInfo(BaseModel):
    name: str
    tracking_uri: str
    description: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame | np.ndarray
    X_test: pd.DataFrame | np.ndarray
    y_train: pd.Series | np.ndarray
    y_test: pd.Series | np.ndarray
