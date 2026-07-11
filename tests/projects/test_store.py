from __future__ import annotations

import json

import pytest

from mlops_toolbox.core.exceptions import ProjectAlreadyRegisteredError, ProjectNotFoundError
from mlops_toolbox.projects import get_project, list_projects, register_project, unregister_project


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
    assert info.tracking_uri == "sqlite:///churn.db"
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
    assert get_project("a", registry_path=registry_path).tracking_uri == "sqlite:///a2.db"


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
    assert raw["projects"]["a"]["tracking_uri"] == "sqlite:///a.db"
