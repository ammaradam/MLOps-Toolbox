from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from mlops_toolbox.core.models import RunResult


class ExperimentTracker(ABC):
    @abstractmethod
    def start_run(
        self, run_name: str | None = None, tags: dict[str, str] | None = None
    ) -> str: ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    @abstractmethod
    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None: ...

    @abstractmethod
    def end_run(self, status: Literal["FINISHED", "FAILED"] = "FINISHED") -> RunResult: ...

    def __enter__(self) -> Self:
        self.start_run()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.end_run(status="FAILED" if exc_type else "FINISHED")
