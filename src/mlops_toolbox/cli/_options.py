"""Option annotations shared by every `mt` subcommand."""

from __future__ import annotations

from typing import Annotated

import typer

TrackingUri = Annotated[
    str, typer.Option("--tracking-uri", help="Tracking store URI.", show_default=True)
]
ModelVersion = Annotated[
    str,
    typer.Option(
        "--version", help="Model version number, 'latest', or an alias (e.g. 'production')."
    ),
]
AssumeYes = Annotated[
    bool, typer.Option("--yes", "-y", help="Skip confirmation prompts (for scripts/CI).")
]

DEFAULT_TRACKING_URI = "sqlite:///mlops.db"


def confirm_or_abort(message: str, assume_yes: bool) -> None:
    """Prompt before a destructive action; `--yes` skips the prompt."""
    if assume_yes:
        return
    if not typer.confirm(message):
        typer.echo("Aborted.")
        raise typer.Exit(code=1)
