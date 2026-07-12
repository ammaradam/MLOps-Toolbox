from __future__ import annotations

from typing import Annotated

import typer

from mlops_toolbox.cli._options import TrackingUri, resolve_tracking_uri


def models(
    project: Annotated[
        str | None,
        typer.Argument(help="Registered project name to list models for (see mt projects)."),
    ] = None,
    tracking_uri: TrackingUri = None,
) -> None:
    """List registered models in a tracking store: version, aliases, contract coverage."""
    from mlops_toolbox.dashboard.data_access import list_registered_models
    from mlops_toolbox.projects import get_project

    uri = (
        get_project(project).tracking_uri
        if project is not None
        else resolve_tracking_uri(tracking_uri)
    )
    infos = list_registered_models(uri)
    if not infos:
        typer.echo(f"No registered models in '{uri}'.")
        raise typer.Exit(code=0)

    typer.echo(f"Models in '{uri}':")
    for info in infos:
        aliases = ", ".join(info.aliases) if info.aliases else "-"
        contract = "yes" if info.tags.get("mlops_toolbox.has_contract") == "true" else "no"
        reference = "yes" if info.tags.get("mlops_toolbox.has_reference") == "true" else "no"
        typer.echo(
            f"  {info.name}  v{info.version}  aliases: {aliases}  "
            f"contract: {contract}  drift baseline: {reference}"
        )


def projects() -> None:
    """List locally registered projects (name, tracking URI, description)."""
    from mlops_toolbox.projects import list_projects

    infos = list_projects()
    if not infos:
        typer.echo(
            "No projects registered yet. Register one from Python: "
            'mt.register_project("my-project", "sqlite:///mlops.db")'
        )
        raise typer.Exit(code=0)

    typer.echo("Registered projects:")
    for info in infos:
        description = f"  — {info.description}" if info.description else ""
        tags = "  " + " ".join(f"[{k}={v}]" for k, v in sorted(info.tags.items()))
        typer.echo(f"  {info.name}  {info.tracking_uri}{description}{tags.rstrip()}")
