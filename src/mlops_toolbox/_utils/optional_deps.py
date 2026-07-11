from __future__ import annotations

import importlib
from types import ModuleType

from mlops_toolbox.core.exceptions import BackendNotInstalledError


def import_optional_dependency(module_name: str, *, extra: str) -> ModuleType:
    """Import a heavy, optional backend dependency.

    Raises BackendNotInstalledError with a `pip install mlops-toolbox[extra]`
    hint instead of letting a bare ImportError surface.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendNotInstalledError(package=module_name, extra=extra) from exc
