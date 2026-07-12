"""FastAPI + Jinja2 dashboard app.

NOTE: this module intentionally avoids `from __future__ import annotations`, for the
same reason as `deployment/fastapi_backend.py`: route handlers are annotated with
types resolved from a dynamically-imported `fastapi` module, and FastAPI needs real
(non-string) annotation objects to introspect them correctly.
"""

from pathlib import Path
from typing import Any

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.exceptions import (
    ProjectAlreadyRegisteredError,
    ProjectNotFoundError,
)
from mlops_toolbox.dashboard.data_access import (
    RunSummary,
    build_metric_series,
    list_registered_models,
    list_runs,
)
from mlops_toolbox.dashboard.sparkline import render_sparkline
from mlops_toolbox.projects import (
    get_project,
    list_projects,
    unregister_project,
    update_project,
)
from mlops_toolbox.projects.health import ProjectHealth, audit_project

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"


def create_app(registry_path: Path | str | None = None) -> Any:
    fastapi_module = import_optional_dependency("fastapi", extra="dashboard")
    import_optional_dependency("jinja2", extra="dashboard")  # used implicitly by Jinja2Templates
    staticfiles = import_optional_dependency("fastapi.staticfiles", extra="dashboard")
    templating = import_optional_dependency("fastapi.templating", extra="dashboard")
    responses = import_optional_dependency("fastapi.responses", extra="dashboard")

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
        summaries: dict[str, dict[str, Any]] = {}
        for project in projects:
            summary: dict[str, Any] = {"health": None, "models": [], "latest_run": None}
            try:
                summary["health"] = audit_project(project.tracking_uri)
                summary["models"] = list_registered_models(project.tracking_uri)
                runs: list[RunSummary] = list_runs(project.tracking_uri, max_results=1)
                summary["latest_run"] = runs[0] if runs else None
            except Exception:
                pass
            summaries[project.name] = summary
        return templates.TemplateResponse(
            request, "index.html", {"projects": projects, "summaries": summaries}
        )

    @app.post("/projects/{name}/edit", name="project_edit")  # type: ignore[untyped-decorator]
    def project_edit(
        request: Request,  # type: ignore[valid-type]
        name: str,
        new_name: str = fastapi_module.Form(""),
        description: str = fastapi_module.Form(""),
    ) -> Any:
        try:
            updated = update_project(
                name,
                new_name=new_name.strip() or None,
                description=description.strip() or None,
                registry_path=app.state.registry_path,
            )
        except ProjectNotFoundError as exc:
            raise fastapi_module.HTTPException(status_code=404, detail=str(exc)) from exc
        except ProjectAlreadyRegisteredError as exc:
            raise fastapi_module.HTTPException(status_code=409, detail=str(exc)) from exc
        return responses.RedirectResponse(
            request.url_for("project_detail", name=updated.name),  # type: ignore[attr-defined]
            status_code=303,
        )

    @app.post("/projects/{name}/delete", name="project_delete")  # type: ignore[untyped-decorator]
    def project_delete(request: Request, name: str) -> Any:  # type: ignore[valid-type]
        try:
            unregister_project(name, registry_path=app.state.registry_path)
        except ProjectNotFoundError as exc:
            raise fastapi_module.HTTPException(status_code=404, detail=str(exc)) from exc
        return responses.RedirectResponse(
            request.url_for("index"),  # type: ignore[attr-defined]
            status_code=303,
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
