from __future__ import annotations

import json

import pytest

from mlops_toolbox.core.exceptions import ProjectAlreadyRegisteredError, ProjectNotFoundError
from mlops_toolbox.projects import (
    get_project,
    list_projects,
    register_project,
    unregister_project,
    update_project,
)


def test_register_list_get_round_trip(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    register_project(
        "churn-model",
        "sqlite:///churn.db",
        description="Churn prediction",
        tags={"team": "growth"},
        registry_path=registry_path,
    )

    projects = list_projects(registry_path=registry_path)
    assert [p.name for p in projects] == ["churn-model"]

    info = get_project("churn-model", registry_path=registry_path)
    # Relative sqlite paths are normalized to absolute at registration time.
    assert info.tracking_uri.startswith("sqlite:///")
    assert info.tracking_uri.endswith("/churn.db")
    assert info.description == "Churn prediction"
    assert info.tags == {"team": "growth"}


def test_register_duplicate_without_overwrite_raises(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    register_project("a", "sqlite:///a.db", registry_path=registry_path)
    with pytest.raises(ProjectAlreadyRegisteredError):
        register_project("a", "sqlite:///a2.db", registry_path=registry_path)


def test_register_duplicate_with_overwrite_replaces(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    register_project("a", "sqlite:///a.db", registry_path=registry_path)
    register_project("a", "sqlite:///a2.db", overwrite=True, registry_path=registry_path)
    assert get_project("a", registry_path=registry_path).tracking_uri.endswith("/a2.db")


def test_get_and_unregister_unknown_raise(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    with pytest.raises(ProjectNotFoundError):
        get_project("missing", registry_path=registry_path)
    with pytest.raises(ProjectNotFoundError):
        unregister_project("missing", registry_path=registry_path)


def test_unregister_removes_project(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    register_project("a", "sqlite:///a.db", registry_path=registry_path)
    unregister_project("a", registry_path=registry_path)
    assert list_projects(registry_path=registry_path) == []


def test_update_project_renames_and_keeps_other_fields(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    register_project(
        "old-name",
        "sqlite:///a.db",
        description="original",
        tags={"team": "x"},
        registry_path=registry_path,
    )

    updated = update_project("old-name", new_name="new-name", registry_path=registry_path)
    assert updated.name == "new-name"
    assert updated.description == "original"
    assert updated.tags == {"team": "x"}
    assert [p.name for p in list_projects(registry_path=registry_path)] == ["new-name"]
    with pytest.raises(ProjectNotFoundError):
        get_project("old-name", registry_path=registry_path)


def test_update_project_description_only(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    register_project("a", "sqlite:///a.db", description="before", registry_path=registry_path)
    updated = update_project("a", description="after", registry_path=registry_path)
    assert updated.name == "a"
    assert updated.description == "after"


def test_update_project_rejects_rename_collision_and_missing(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    register_project("a", "sqlite:///a.db", registry_path=registry_path)
    register_project("b", "sqlite:///b.db", registry_path=registry_path)
    with pytest.raises(ProjectAlreadyRegisteredError):
        update_project("a", new_name="b", registry_path=registry_path)
    with pytest.raises(ProjectNotFoundError):
        update_project("missing", new_name="c", registry_path=registry_path)


def test_list_projects_on_missing_file_returns_empty(tmp_path) -> None:
    registry_path = tmp_path / "does-not-exist.json"
    assert list_projects(registry_path=registry_path) == []


def test_mlops_toolbox_home_env_var_is_respected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLOPS_TOOLBOX_HOME", str(tmp_path))
    register_project("a", "sqlite:///a.db")
    assert (tmp_path / "projects.json").exists()
    assert [p.name for p in list_projects()] == ["a"]


def test_registry_file_shape(tmp_path) -> None:
    registry_path = tmp_path / "projects.json"
    register_project("a", "sqlite:///a.db", registry_path=registry_path)
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert "a" in raw["projects"]
    assert raw["projects"]["a"]["tracking_uri"].endswith("/a.db")
