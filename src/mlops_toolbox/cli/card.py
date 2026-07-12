from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_toolbox.cli._options import ModelVersion, TrackingUri


def card(
    name: Annotated[str, typer.Argument(help="Registered model name.")],
    version: ModelVersion = "latest",
    tracking_uri: TrackingUri = None,
    output_path: Annotated[
        Path | None,
        typer.Option("--out", help="Write the card here instead of printing it."),
    ] = None,
) -> None:
    """Render a markdown model card from the registry: schema, metrics, monitoring status."""
    from mlops_toolbox.cards import generate_model_card
    from mlops_toolbox.factories import model_registry

    markdown = generate_model_card(model_registry(tracking_uri), name, version)
    if output_path is None:
        typer.echo(markdown)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    typer.echo(f"Wrote model card for '{name}' ({version}) to {output_path}")
