from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.models import ModelInfo


class RunSummary(BaseModel):
    run_id: str
    run_name: str | None
    status: str
    start_time: datetime | None
    end_time: datetime | None
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)


def _to_datetime(millis: int | None) -> datetime | None:
    if millis is None:
        return None
    return datetime.fromtimestamp(millis / 1000, tz=UTC)


def list_runs(tracking_uri: str, max_results: int = 50) -> list[RunSummary]:
    """Newest-first run summaries across every experiment in this tracking store."""
    mlflow = import_optional_dependency("mlflow", extra="tracking")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    experiment_ids = [exp.experiment_id for exp in client.search_experiments()]
    if not experiment_ids:
        return []

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        order_by=["start_time DESC"],
        max_results=max_results,
    )
    return [
        RunSummary(
            run_id=run.info.run_id,
            run_name=run.info.run_name or None,
            status=run.info.status,
            start_time=_to_datetime(run.info.start_time),
            end_time=_to_datetime(run.info.end_time),
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
        )
        for run in runs
    ]


def list_registered_models(tracking_uri: str) -> list[ModelInfo]:
    """One entry per registered model name, resolved to its highest version number.

    Mirrors the "latest" resolution `registry/mlflow_backend.py` already uses, so the
    dashboard and the registry agree on what "latest" means.
    """
    mlflow = import_optional_dependency("mlflow", extra="tracking")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    infos: list[ModelInfo] = []
    for registered_model in client.search_registered_models():
        versions = client.search_model_versions(f"name='{registered_model.name}'")
        if not versions:
            continue
        latest = max(versions, key=lambda v: int(v.version))
        # Re-fetch by version: search results omit aliases in some stores.
        refreshed = client.get_model_version(registered_model.name, latest.version)
        infos.append(_to_model_info(refreshed))
    return infos


def _to_model_info(mv: Any) -> ModelInfo:
    return ModelInfo(
        name=mv.name,
        version=str(mv.version),
        backend="mlflow",
        aliases=sorted(mv.aliases) if mv.aliases else [],
        source_run_id=mv.run_id,
        uri=f"models:/{mv.name}/{mv.version}",
        tags=dict(mv.tags) if mv.tags else {},
    )


def build_metric_series(runs: list[RunSummary]) -> dict[str, list[tuple[datetime, float]]]:
    """Pivot per-run final metric values into per-metric-key series, oldest run first."""
    chronological = sorted(runs, key=lambda r: r.start_time or datetime.min.replace(tzinfo=UTC))

    series: dict[str, list[tuple[datetime, float]]] = {}
    for run in chronological:
        if run.start_time is None:
            continue
        for key, value in run.metrics.items():
            series.setdefault(key, []).append((run.start_time, value))
    return series
