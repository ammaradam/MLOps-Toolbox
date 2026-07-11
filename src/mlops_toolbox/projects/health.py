"""Lifecycle-completeness audit: which stages has a project actually covered?

Each failing check's hint names the toolbox call that closes the gap, so the
audit doubles as a checklist of what a production-ready model needs. The audit
is strictly read-only — it never creates experiments or writes to the store.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.models import ModelContract, ModelInfo

_HAS_CONTRACT_TAG = "mlops_toolbox.has_contract"
_HAS_REFERENCE_TAG = "mlops_toolbox.has_reference"


class HealthCheck(BaseModel):
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str
    hint: str | None = None


class ProjectHealth(BaseModel):
    tracking_uri: str
    checks: list[HealthCheck] = Field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(1 for check in self.checks if check.status == "pass")

    @property
    def n_warn(self) -> int:
        return sum(1 for check in self.checks if check.status == "warn")

    @property
    def n_fail(self) -> int:
        return sum(1 for check in self.checks if check.status == "fail")

    @property
    def passed(self) -> bool:
        return self.n_fail == 0


def audit_project(tracking_uri: str) -> ProjectHealth:
    """Run the ordered lifecycle checks against one tracking store."""
    from mlops_toolbox.dashboard.data_access import list_registered_models, list_runs

    checks: list[HealthCheck] = []

    try:
        runs = list_runs(tracking_uri)
    except Exception as exc:
        checks.append(
            HealthCheck(
                name="tracking store reachable",
                status="fail",
                detail=str(exc),
                hint="Check the tracking URI (e.g. sqlite:///mlops.db) and that it exists.",
            )
        )
        return ProjectHealth(tracking_uri=tracking_uri, checks=checks)
    checks.append(
        HealthCheck(name="tracking store reachable", status="pass", detail=tracking_uri)
    )

    if runs:
        checks.append(
            HealthCheck(
                name="experiment runs recorded", status="pass", detail=f"{len(runs)} run(s)"
            )
        )
    else:
        checks.append(
            HealthCheck(
                name="experiment runs recorded",
                status="fail",
                detail="no runs found",
                hint="Track a run: `with mt.tracker(tracking_uri=...) as run_tracker: ...`",
            )
        )

    if runs:
        n_with_metrics = sum(1 for run in runs if run.metrics)
        if n_with_metrics:
            checks.append(
                HealthCheck(
                    name="runs log metrics",
                    status="pass",
                    detail=f"{n_with_metrics} of {len(runs)} run(s) have metrics",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    name="runs log metrics",
                    status="warn",
                    detail="no run has metrics",
                    hint="Log evaluation results: `tracker.log_metrics(report.metrics)`",
                )
            )

    models = list_registered_models(tracking_uri)
    if models:
        names = ", ".join(model.name for model in models)
        checks.append(
            HealthCheck(name="model registered", status="pass", detail=names)
        )
        checks.extend(_contract_checks(tracking_uri, models))
    else:
        checks.append(
            HealthCheck(
                name="model registered",
                status="fail",
                detail="no registered models",
                hint="Register one: `registry.register_model(mt.adapt(model), name=...)`",
            )
        )

    return ProjectHealth(tracking_uri=tracking_uri, checks=checks)


def _contract_checks(tracking_uri: str, models: list[ModelInfo]) -> list[HealthCheck]:
    checks: list[HealthCheck] = []

    with_contract = [m for m in models if m.tags.get(_HAS_CONTRACT_TAG) == "true"]
    checks.append(
        _coverage_check(
            name="model carries a contract",
            covered=len(with_contract),
            total=len(models),
            hint=(
                "Store the input schema at registration: "
                "`register_model(..., contract=mt.infer_contract(X_train, evaluation=report))`"
            ),
        )
    )

    with_reference = [m for m in models if m.tags.get(_HAS_REFERENCE_TAG) == "true"]
    checks.append(
        _coverage_check(
            name="drift reference stored",
            covered=len(with_reference),
            total=len(models),
            hint=(
                "Store a drift baseline at registration: "
                "`register_model(..., reference_data=X_train)` — then `mt.check_drift(...)` works."
            ),
        )
    )

    if with_contract:
        with_evaluation = [
            m for m in with_contract if _contract_has_evaluation(tracking_uri, m)
        ]
        if len(with_evaluation) == len(with_contract):
            status: Literal["pass", "warn"] = "pass"
        else:
            status = "warn"
        checks.append(
            HealthCheck(
                name="contract records evaluation",
                status=status,
                detail=f"{len(with_evaluation)} of {len(with_contract)} contract(s) include one",
                hint=None
                if status == "pass"
                else "Pass the eval report: `mt.infer_contract(X_train, evaluation=report)`",
            )
        )

    return checks


def _coverage_check(name: str, covered: int, total: int, hint: str) -> HealthCheck:
    if covered == total:
        return HealthCheck(name=name, status="pass", detail=f"{covered} of {total} model(s)")
    status: Literal["warn", "fail"] = "warn" if covered else "fail"
    return HealthCheck(
        name=name, status=status, detail=f"{covered} of {total} model(s)", hint=hint
    )


def _contract_has_evaluation(tracking_uri: str, model: ModelInfo) -> bool:
    """Read-only contract download via MlflowClient (no registry side effects)."""
    if model.source_run_id is None:
        return False
    mlflow = import_optional_dependency("mlflow", extra="tracking")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = client.download_artifacts(model.source_run_id, "contract/contract.json", tmp)
        except Exception:
            return False
        contract = ModelContract.model_validate_json(Path(local).read_text(encoding="utf-8"))
    return contract.evaluation is not None
