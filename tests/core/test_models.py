from __future__ import annotations

import numpy as np

from mlops_toolbox.core.models import (
    ColumnDriftResult,
    DatasetSplit,
    DriftReport,
    EvaluationReport,
    ModelInfo,
    RunResult,
    ValidationResult,
)


def test_run_result_defaults() -> None:
    run = RunResult(run_id="abc123", backend="mlflow")
    assert run.status == "RUNNING"
    assert run.params == {}
    assert run.metrics == {}
    assert run.end_time is None


def test_model_info_round_trip() -> None:
    info = ModelInfo(name="churn-model", version="1", backend="mlflow", uri="models:/churn-model/1")
    assert info.aliases == []
    assert info.tags == {}


def test_evaluation_report_classification() -> None:
    report = EvaluationReport(
        task_type="classification",
        metrics={"accuracy": 0.9},
        n_samples=10,
        confusion_matrix=[[5, 0], [1, 4]],
    )
    assert report.metrics["accuracy"] == 0.9


def test_drift_report_with_columns() -> None:
    report = DriftReport(
        dataset_drift_detected=True,
        drift_share=0.5,
        column_results={
            "col_a": ColumnDriftResult(
                column="col_a", drift_detected=True, drift_score=0.9, stattest="ks"
            )
        },
    )
    assert report.column_results["col_a"].drift_detected is True


def test_validation_result() -> None:
    result = ValidationResult(is_valid=False, errors=["row 0: bad"], schema_name="RowSchema")
    assert result.is_valid is False
    assert len(result.errors) == 1


def test_dataset_split_holds_arrays() -> None:
    split = DatasetSplit(
        X_train=np.zeros((2, 2)),
        X_test=np.zeros((1, 2)),
        y_train=np.zeros(2),
        y_test=np.zeros(1),
    )
    assert split.X_train.shape == (2, 2)
