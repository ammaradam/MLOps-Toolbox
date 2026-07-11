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

DEFAULT_TRACKING_URI = "sqlite:///mlops.db"
