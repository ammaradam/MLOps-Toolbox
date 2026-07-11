from __future__ import annotations

import pytest

import mlops_toolbox as mt
from mlops_toolbox.core.exceptions import BackendNotInstalledError


def test_all_public_names_resolve() -> None:
    for name in mt.__all__:
        if name == "__version__":
            assert isinstance(mt.__version__, str)
            continue
        try:
            attr = getattr(mt, name)
        except BackendNotInstalledError:
            pytest.skip(f"optional backend for {name} not installed")
            continue
        assert attr is not None


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = mt.NotARealExport
