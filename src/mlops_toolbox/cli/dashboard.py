from __future__ import annotations

from typing import Annotated

import typer


def dashboard(
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8050,
) -> None:
    """Run the multi-project dashboard (read-only) over all registered projects."""
    from mlops_toolbox.dashboard.app import run_dashboard
    from mlops_toolbox.projects import list_projects

    n_projects = len(list_projects())
    if n_projects == 0:
        typer.echo(
            "No projects registered yet — the dashboard will be empty. "
            "Register one first: mt register <name> --tracking-uri <uri>"
        )
    typer.echo(f"Dashboard on http://{host}:{port} ({n_projects} project(s))")
    run_dashboard(host=host, port=port)
