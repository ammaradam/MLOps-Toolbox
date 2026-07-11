from __future__ import annotations

import pytest

from mlops_toolbox.core.models import EvaluationReport
from mlops_toolbox.evaluation.gates import check_gates, parse_gate


def test_parse_gate_valid_expressions() -> None:
    assert parse_gate("f1_macro>=0.85") == ("f1_macro", ">=", 0.85)
    assert parse_gate("accuracy > 0.9") == ("accuracy", ">", 0.9)
    assert parse_gate("mae<=2") == ("mae", "<=", 2.0)
    assert parse_gate("bias == -0.5") == ("bias", "==", -0.5)


@pytest.mark.parametrize("expression", ["", "f1", "f1 => 0.8", "0.8 <= f1", "f1 >= high"])
def test_parse_gate_invalid_expressions(expression: str) -> None:
    with pytest.raises(ValueError, match="Invalid gate expression"):
        parse_gate(expression)


def test_check_gates_pass_and_fail() -> None:
    metrics = {"accuracy": 0.92, "f1_macro": 0.80}
    result = check_gates(metrics, ["accuracy>=0.9", "f1_macro>=0.85"])
    assert result.passed is False
    by_metric = {check.metric: check for check in result.checks}
    assert by_metric["accuracy"].passed is True
    assert by_metric["accuracy"].actual == 0.92
    assert by_metric["f1_macro"].passed is False


def test_check_gates_all_pass() -> None:
    result = check_gates({"mae": 1.5}, ["mae<=2", "mae>1"])
    assert result.passed is True


def test_check_gates_missing_metric_fails_with_hint() -> None:
    result = check_gates({"accuracy": 0.9}, ["f1_macro>=0.5"])
    assert result.passed is False
    assert "not found" in result.checks[0].message
    assert "accuracy" in result.checks[0].message


def test_evaluation_report_check_gates() -> None:
    report = EvaluationReport(task_type="classification", metrics={"accuracy": 0.95}, n_samples=10)
    assert report.check_gates(["accuracy>=0.9"]).passed is True
    assert report.check_gates(["accuracy>=0.99"]).passed is False
