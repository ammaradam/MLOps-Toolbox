from __future__ import annotations

import pytest

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.exceptions import BackendNotInstalledError


def test_missing_dependency_raises_friendly_error() -> None:
    with pytest.raises(BackendNotInstalledError) as excinfo:
        import_optional_dependency("definitely_not_a_real_package", extra="fake-extra")
    assert "pip install 'mlops-toolbox[fake-extra]'" in str(excinfo.value)


def test_installed_dependency_returns_module() -> None:
    module = import_optional_dependency("json", extra="n/a")
    assert module.__name__ == "json"
