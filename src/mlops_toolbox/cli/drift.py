from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_toolbox.cli._io import read_frame
from mlops_toolbox.cli._options import ModelVersion, TrackingUri


def drift(
    name: Annotated[str, typer.Argument(help="Registered model name.")],
    input_path: Annotated[
        Path, typer.Option("--input", help="Current data to check (.parquet or .csv).")
    ],
    version: ModelVersion = "latest",
    tracking_uri: TrackingUri = None,
    predictions: Annotated[
        bool,
        typer.Option(
            "--predictions",
            help="Treat the input's first column as predictions and check prediction drift "
            "instead of input-data drift.",
        ),
    ] = False,
) -> None:
    """Check drift against the baseline stored with the model; exits 1 when drift is detected."""
    from mlops_toolbox.core.exceptions import MonitoringError
    from mlops_toolbox.factories import model_registry
    from mlops_toolbox.monitoring import check_drift, check_prediction_drift

    current = read_frame(input_path)
    registry = model_registry(tracking_uri)
    try:
        if predictions:
            report = check_prediction_drift(
                current.iloc[:, 0], registry=registry, name=name, version=version
            )
        else:
            report = check_drift(current, registry=registry, name=name, version=version)
    except MonitoringError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    kind = "prediction" if predictions else "input-data"
    typer.echo(f"Checking {kind} drift for '{name}' ({version}) against stored baseline")
    for column, result in sorted(report.column_results.items()):
        status = "DRIFT" if result.drift_detected else "ok"
        typer.echo(f"  [{status:5}] {column}: score={result.drift_score:.3f} ({result.stattest})")
    share = f"{report.drift_share:.0%} of columns drifted"
    if report.dataset_drift_detected:
        typer.echo(f"Drift DETECTED ({share}).")
        raise typer.Exit(code=1)
    typer.echo(f"No dataset drift ({share}).")
