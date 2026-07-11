"""FastAPI + Jinja2 dashboard app.

NOTE: this module intentionally avoids `from __future__ import annotations`, for the
same reason as `deployment/fastapi_backend.py`: route handlers are annotated with
types resolved from a dynamically-imported `fastapi` module, and FastAPI needs real
(non-string) annotation objects to introspect them correctly.
"""

from pathlib import Path
from typing import Any

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.exceptions import ProjectNotFoundError
from mlops_toolbox.dashboard.data_access import (
    build_metric_series,
    list_registered_models,
    list_runs,
)
from mlops_toolbox.dashboard.sparkline import render_sparkline
from mlops_toolbox.projects import get_project, list_projects
from mlops_toolbox.projects.health import ProjectHealth, audit_project

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"


def create_app(registry_path: Path | str | None = None) -> Any:
    fastapi_module = import_optional_dependency("fastapi", extra="dashboard")
    import_optional_dependency("jinja2", extra="dashboard")  # used implicitly by Jinja2Templates
    staticfiles = import_optional_dependency("fastapi.staticfiles", extra="dashboard")
    templating = import_optional_dependency("fastapi.templating", extra="dashboard")

    app = fastapi_module.FastAPI(title="mlops-toolbox dashboard")
    app.state.registry_path = registry_path
    templates = templating.Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app.mount("/static", staticfiles.StaticFiles(directory=str(_STATIC_DIR)), name="static")

    Request = fastapi_module.Request

    @app.get("/health")  # type: ignore[untyped-decorator]
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", name="index")  # type: ignore[untyped-decorator]
    def index(request: Request) -> Any:  # type: ignore[valid-type]
        projects = list_projects(registry_path=app.state.registry_path)
        health_by_name: dict[str, ProjectHealth | None] = {}
        for project in projects:
            try:
                health_by_name[project.name] = audit_project(project.tracking_uri)
            except Exception:
                health_by_name[project.name] = None
        return templates.TemplateResponse(
            request, "index.html", {"projects": projects, "health_by_name": health_by_name}
        )

    @app.get("/projects/{name}", name="project_detail")  # type: ignore[untyped-decorator]
    def project_detail(request: Request, name: str) -> Any:  # type: ignore[valid-type]
        try:
            project = get_project(name, registry_path=app.state.registry_path)
        except ProjectNotFoundError as exc:
            raise fastapi_module.HTTPException(status_code=404, detail=str(exc)) from exc

        runs = None
        runs_error = None
        stat_tiles: list[dict[str, Any]] = []
        try:
            runs = list_runs(project.tracking_uri)
            for metric_name, points in build_metric_series(runs).items():
                values = [value for _, value in points]
                stat_tiles.append(
                    {"name": metric_name, "latest": values[-1], "svg": render_sparkline(values)}
                )
        except Exception as exc:
            runs_error = str(exc)

        models = None
        models_error = None
        try:
            models = list_registered_models(project.tracking_uri)
        except Exception as exc:
            models_error = str(exc)

        health: ProjectHealth | None = None
        try:
            health = audit_project(project.tracking_uri)
        except Exception:
            health = None

        return templates.TemplateResponse(
            request,
            "project_detail.html",
            {
                "project": project,
                "runs": runs,
                "runs_error": runs_error,
                "models": models,
                "models_error": models_error,
                "stat_tiles": stat_tiles,
                "health": health,
            },
        )

    return app


def run_dashboard(
    host: str = "127.0.0.1", port: int = 8050, registry_path: Path | str | None = None
) -> None:
    uvicorn = import_optional_dependency("uvicorn", extra="dashboard")
    app = create_app(registry_path=registry_path)
    uvicorn.run(app, host=host, port=port)
