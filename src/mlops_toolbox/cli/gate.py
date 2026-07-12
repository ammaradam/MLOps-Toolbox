from __future__ import annotations

from typing import Annotated

import typer

from mlops_toolbox.cli._options import ModelVersion, TrackingUri, resolve_tracking_uri


def _resolve_metrics(
    tracking_uri: str, run_id: str | None, model: str | None, version: str
) -> tuple[str, dict[str, float]]:
    """Return (metrics source description, final metric values) for the gated run."""
    from mlops_toolbox._utils.optional_deps import import_optional_dependency

    mlflow = import_optional_dependency("mlflow", extra="tracking")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    if model is not None:
        from mlops_toolbox.factories import model_registry

        info = model_registry(tracking_uri).get_model_info(model, version)
        if info.source_run_id is None:
            raise typer.BadParameter(f"Model '{model}' v{info.version} has no source run.")
        run = client.get_run(info.source_run_id)
        return f"model '{model}' v{info.version} (run {info.source_run_id})", dict(
            run.data.metrics
        )

    if run_id is not None:
        run = client.get_run(run_id)
        return f"run {run_id}", dict(run.data.metrics)

    from mlops_toolbox.dashboard.data_access import list_runs

    for summary in list_runs(tracking_uri):
        if summary.status == "FINISHED" and summary.metrics:
            return f"latest finished run {summary.run_id}", summary.metrics
    raise typer.BadParameter(
        f"No finished run with metrics found in '{tracking_uri}'. "
        "Pass --run or --model to select one explicitly."
    )


def gate(
    expressions: Annotated[
        list[str], typer.Argument(help='Gate expressions, e.g. "f1_macro>=0.85".')
    ],
    run_id: Annotated[
        str | None, typer.Option("--run", help="Gate this run's metrics.")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Gate the source run of this registered model.")
    ] = None,
    version: ModelVersion = "latest",
    tracking_uri: TrackingUri = None,
) -> None:
    """Check metric gates against an MLflow run; exits 1 when any gate fails (CI-friendly)."""
    from mlops_toolbox.evaluation.gates import check_gates

    if run_id is not None and model is not None:
        raise typer.BadParameter("Pass either --run or --model, not both.")

    source, metrics = _resolve_metrics(resolve_tracking_uri(tracking_uri), run_id, model, version)
    result = check_gates(metrics, expressions)

    typer.echo(f"Gating {source}")
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        typer.echo(f"  [{status}] {check.expression}: {check.message}")
    typer.echo("All gates passed." if result.passed else "Gate check FAILED.")
    raise typer.Exit(code=0 if result.passed else 1)
