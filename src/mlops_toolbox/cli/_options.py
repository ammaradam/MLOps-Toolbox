"""Option annotations shared by every `mt` subcommand."""

from __future__ import annotations

from typing import Annotated

import typer

TrackingUri = Annotated[
    str | None,
    typer.Option(
        "--tracking-uri",
        help="Tracking store URI (default: mlops.toml / MT_TRACKING_URI, else sqlite:///mlops.db).",
        show_default=False,
    ),
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


def resolve_tracking_uri(tracking_uri: str | None) -> str:
    """Resolve an omitted --tracking-uri through settings (env > mlops.toml > default)."""
    if tracking_uri is not None:
        return tracking_uri
    from mlops_toolbox.settings import load_settings

    return load_settings().tracking_uri


def confirm_or_abort(message: str, assume_yes: bool) -> None:
    """Prompt before a destructive action; `--yes` skips the prompt."""
    if assume_yes:
        return
    if not typer.confirm(message):
        typer.echo("Aborted.")
        raise typer.Exit(code=1)
