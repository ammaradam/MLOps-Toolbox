"""Metric gates: threshold expressions checked against evaluation metrics.

The CI-facing half of evaluation — `mt gate "f1_macro>=0.85"` turns model
quality into a passing or failing check.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Sequence

from mlops_toolbox.core.models import GateCheck, GateResult

_GATE_RE = re.compile(
    r"^\s*(?P<metric>[A-Za-z_][\w.\-]*)\s*(?P<op>>=|<=|==|>|<)\s*(?P<threshold>-?\d+(?:\.\d+)?)\s*$"
)

_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}


def parse_gate(expression: str) -> tuple[str, str, float]:
    """Parse "metric >= threshold" into (metric, op, threshold); raises ValueError."""
    match = _GATE_RE.match(expression)
    if match is None:
        raise ValueError(
            f"Invalid gate expression {expression!r}. Expected '<metric> <op> <number>' with "
            "op one of >=, <=, ==, >, < (e.g. 'f1_macro>=0.85')."
        )
    return match["metric"], match["op"], float(match["threshold"])


def check_gates(metrics: dict[str, float], gates: Sequence[str]) -> GateResult:
    """Check every gate expression against `metrics`; overall pass requires all to pass."""
    checks: list[GateCheck] = []
    for expression in gates:
        metric, op, threshold = parse_gate(expression)
        actual = metrics.get(metric)
        if actual is None:
            available = ", ".join(sorted(metrics)) or "(none)"
            checks.append(
                GateCheck(
                    expression=expression,
                    metric=metric,
                    passed=False,
                    message=f"metric '{metric}' not found; available: {available}",
                )
            )
            continue
        passed = _OPERATORS[op](actual, threshold)
        checks.append(
            GateCheck(
                expression=expression,
                metric=metric,
                passed=passed,
                actual=actual,
                message=f"{metric} = {actual:.6g} (required {op} {threshold:g})",
            )
        )
    return GateResult(passed=all(check.passed for check in checks), checks=checks)
