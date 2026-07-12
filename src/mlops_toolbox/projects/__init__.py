from __future__ import annotations

from pathlib import Path

from mlops_toolbox.core.exceptions import ProjectAlreadyRegisteredError, ProjectNotFoundError
from mlops_toolbox.core.models import ProjectInfo
from mlops_toolbox.projects.store import read_projects, resolve_registry_path, write_projects

__all__ = [
    "register_project",
    "list_projects",
    "get_project",
    "update_project",
    "unregister_project",
]

_SQLITE_PREFIX = "sqlite:///"


def _normalize_tracking_uri(tracking_uri: str) -> str:
    """Resolve relative sqlite paths to absolute ones at registration time.

    A relative sqlite URI resolves against the CWD of whoever *reads* it (and
    the backend silently creates an empty store when the file is missing), so
    storing one in the project registry is almost never what the user means.
    """
    if not tracking_uri.startswith(_SQLITE_PREFIX):
        return tracking_uri
    path = Path(tracking_uri[len(_SQLITE_PREFIX) :])
    if path.is_absolute():
        return tracking_uri
    return _SQLITE_PREFIX + (Path.cwd() / path).resolve().as_posix()


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
        name=name,
        tracking_uri=_normalize_tracking_uri(tracking_uri),
        description=description,
        tags=tags or {},
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


def update_project(
    name: str,
    *,
    new_name: str | None = None,
    description: str | None = None,
    registry_path: Path | str | None = None,
) -> ProjectInfo:
    """Rename a project and/or replace its description; other fields are kept."""
    path = resolve_registry_path(registry_path)
    projects = read_projects(path)
    if name not in projects:
        raise ProjectNotFoundError(name)
    if new_name and new_name != name and new_name in projects:
        raise ProjectAlreadyRegisteredError(new_name)

    info = projects.pop(name)
    updated = info.model_copy(
        update={
            "name": new_name or info.name,
            "description": description if description is not None else info.description,
        }
    )
    projects[updated.name] = updated
    write_projects(path, projects)
    return updated


def unregister_project(name: str, *, registry_path: Path | str | None = None) -> None:
    path = resolve_registry_path(registry_path)
    projects = read_projects(path)
    if name not in projects:
        raise ProjectNotFoundError(name)
    del projects[name]
    write_projects(path, projects)
