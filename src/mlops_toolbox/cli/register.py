from __future__ import annotations

from typing import Annotated

import typer

from mlops_toolbox.cli._options import (
    DEFAULT_TRACKING_URI,
    AssumeYes,
    TrackingUri,
    confirm_or_abort,
)


def _parse_tags(raw_tags: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for raw in raw_tags:
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise typer.BadParameter(f"Tag {raw!r} must be key=value.")
        tags[key] = value
    return tags


def register(
    name: Annotated[str, typer.Argument(help="Project name.")],
    tracking_uri: TrackingUri = DEFAULT_TRACKING_URI,
    description: Annotated[
        str | None, typer.Option("--description", help="Short project description.")
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Project tag as key=value; repeatable."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing registration.")
    ] = False,
    yes: AssumeYes = False,
) -> None:
    """Register a project (name -> tracking URI) for the dashboard, doctor, and listings.

    Relative sqlite paths are resolved to absolute ones at registration time,
    so the project points at the same store no matter where commands run later.
    """
    from mlops_toolbox.core.exceptions import ProjectAlreadyRegisteredError, ProjectNotFoundError
    from mlops_toolbox.projects import get_project, register_project

    if overwrite:
        try:
            existing = get_project(name)
        except ProjectNotFoundError:
            existing = None
        if existing is not None:
            confirm_or_abort(
                f"Replace existing registration for '{name}' "
                f"(currently -> {existing.tracking_uri})?",
                yes,
            )

    try:
        info = register_project(
            name,
            tracking_uri,
            description=description,
            tags=_parse_tags(tag or []),
            overwrite=overwrite,
        )
    except ProjectAlreadyRegisteredError as exc:
        typer.echo(f"{exc} (from the CLI: pass --overwrite)", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Registered project '{info.name}' -> {info.tracking_uri}")
    typer.echo(f"Next:  mt doctor {info.name}   or   mt dashboard")


def unregister(
    name: Annotated[str, typer.Argument(help="Project name.")],
    yes: AssumeYes = False,
) -> None:
    """Remove a project registration (the tracking store itself is untouched)."""
    from mlops_toolbox.core.exceptions import ProjectNotFoundError
    from mlops_toolbox.projects import get_project, unregister_project

    try:
        get_project(name)
    except ProjectNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    confirm_or_abort(
        f"Unregister project '{name}'? (The tracking store itself is untouched.)", yes
    )
    unregister_project(name)
    typer.echo(f"Unregistered project '{name}'.")
