from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_toolbox.cli._io import read_frame, write_frame
from mlops_toolbox.cli._options import ModelVersion, TrackingUri


def score(
    name: Annotated[str, typer.Argument(help="Registered model name.")],
    input_path: Annotated[
        Path, typer.Option("--input", help="Data to score (.parquet or .csv).")
    ],
    output_path: Annotated[
        Path | None,
        typer.Option("--out", help="Where to write scored data (default: <input>.scored.<ext>)."),
    ] = None,
    version: ModelVersion = "latest",
    tracking_uri: TrackingUri = None,
    prediction_column: Annotated[
        str, typer.Option("--prediction-column", help="Name of the appended prediction column.")
    ] = "prediction",
) -> None:
    """Batch-score a file with a registered model, enforcing its contract on the input."""
    from mlops_toolbox.core.exceptions import ValidationFailedError
    from mlops_toolbox.factories import model_registry
    from mlops_toolbox.scoring import score_dataframe

    df = read_frame(input_path)
    registry = model_registry(tracking_uri)
    try:
        scored = score_dataframe(
            df,
            registry=registry,
            name=name,
            version=version,
            prediction_column=prediction_column,
        )
    except ValidationFailedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    out = output_path or input_path.with_suffix(f".scored{input_path.suffix}")
    write_frame(scored, out)
    typer.echo(f"Scored {len(scored)} row(s) with '{name}' ({version}) -> {out}")
