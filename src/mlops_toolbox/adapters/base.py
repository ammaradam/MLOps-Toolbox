from __future__ import annotations

from typing import Any


def detect_framework(model: Any) -> str:
    """Best-effort framework label based on the model's defining module.

    Used only for tagging/reporting purposes (e.g. ModelInfo.tags) — it never
    triggers an import of the detected framework.
    """
    module_name = type(model).__module__
    for prefix in ("sklearn", "xgboost", "lightgbm", "catboost", "torch", "tensorflow", "keras"):
        if module_name.startswith(prefix):
            return prefix
    return "custom"
