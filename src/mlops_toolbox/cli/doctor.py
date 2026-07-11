from __future__ import annotations

from typing import Annotated

import typer

from mlops_toolbox.cli._options import DEFAULT_TRACKING_URI


def doctor(
    project: Annotated[
        str | None,
        typer.Argument(help="Registered project name (see mt.register_project)."),
    ] = None,
    tracking_uri: Annotated[
        str | None, typer.Option("--tracking-uri", help="Audit this tracking URI directly.")
    ] = None,
) -> None:
    """Audit a project's lifecycle completeness; exits 1 when any check fails."""
    from mlops_toolbox.core.exceptions import ProjectNotFoundError
    from mlops_toolbox.projects.health import audit_project

    if project is not None and tracking_uri is not None:
        raise typer.BadParameter("Pass either a project name or --tracking-uri, not both.")
    if project is not None:
        from mlops_toolbox.projects import get_project

        try:
            uri = get_project(project).tracking_uri
        except ProjectNotFoundError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
    else:
        uri = tracking_uri or DEFAULT_TRACKING_URI

    health = audit_project(uri)
    typer.echo(f"Auditing {uri}")
    for check in health.checks:
        typer.echo(f"  [{check.status.upper():4}] {check.name}: {check.detail}")
        if check.hint and check.status != "pass":
            typer.echo(f"         hint: {check.hint}")

    summary = f"{health.n_pass} passed, {health.n_warn} warned, {health.n_fail} failed"
    typer.echo(f"Lifecycle health: {summary}.")
    raise typer.Exit(code=0 if health.passed else 1)
