"""`mt` console entry point.

The console script is installed with the base package, so guard the typer
import with the usual extras hint instead of a bare ImportError traceback.
"""

from __future__ import annotations

import sys

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.exceptions import BackendNotInstalledError


def main() -> None:
    try:
        import_optional_dependency("typer", extra="cli")
    except BackendNotInstalledError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    from mlops_toolbox.cli._app import app

    app()
