from __future__ import annotations

import json
import os
from pathlib import Path

from mlops_toolbox.core.models import ProjectInfo

_SCHEMA_VERSION = 1
_ENV_VAR = "MLOPS_TOOLBOX_HOME"
_FILE_NAME = "projects.json"


def resolve_registry_path(registry_path: Path | str | None = None) -> Path:
    if registry_path is not None:
        return Path(registry_path)
    home = os.environ.get(_ENV_VAR)
    base = Path(home) if home else Path.home() / ".mlops-toolbox"
    return base / _FILE_NAME


def read_projects(registry_path: Path) -> dict[str, ProjectInfo]:
    if not registry_path.exists():
        return {}
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return {
        name: ProjectInfo.model_validate(entry) for name, entry in data.get("projects", {}).items()
    }


def write_projects(registry_path: Path, projects: dict[str, ProjectInfo]) -> None:
    """Atomically write the project registry (write-tmp-then-replace, safe on POSIX and Windows)."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _SCHEMA_VERSION,
        "projects": {name: info.model_dump(mode="json") for name, info in projects.items()},
    }
    tmp_path = registry_path.with_name(registry_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, registry_path)
