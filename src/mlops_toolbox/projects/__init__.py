from __future__ import annotations

from pathlib import Path

from mlops_toolbox.core.exceptions import ProjectAlreadyRegisteredError, ProjectNotFoundError
from mlops_toolbox.core.models import ProjectInfo
from mlops_toolbox.projects.store import read_projects, resolve_registry_path, write_projects

__all__ = ["register_project", "list_projects", "get_project", "unregister_project"]


def register_project(
    name: str,
    tracking_uri: str,
    description: str | None = None,
    tags: dict[str, str] | None = None,
    *,
    overwrite: bool = False,
    registry_path: Path | str | None = None,
) -> ProjectInfo:
    path = resolve_registry_path(registry_path)
    projects = read_projects(path)
    if name in projects and not overwrite:
        raise ProjectAlreadyRegisteredError(name)

    info = ProjectInfo(
        name=name, tracking_uri=tracking_uri, description=description, tags=tags or {}
    )
    projects[name] = info
    write_projects(path, projects)
    return info


def list_projects(*, registry_path: Path | str | None = None) -> list[ProjectInfo]:
    path = resolve_registry_path(registry_path)
    return list(read_projects(path).values())


def get_project(name: str, *, registry_path: Path | str | None = None) -> ProjectInfo:
    path = resolve_registry_path(registry_path)
    projects = read_projects(path)
    if name not in projects:
        raise ProjectNotFoundError(name)
    return projects[name]


def unregister_project(name: str, *, registry_path: Path | str | None = None) -> None:
    path = resolve_registry_path(registry_path)
    projects = read_projects(path)
    if name not in projects:
        raise ProjectNotFoundError(name)
    del projects[name]
    write_projects(path, projects)
