class MLOpsToolboxError(Exception):
    """Base exception for all mlops-toolbox errors."""


class BackendNotInstalledError(MLOpsToolboxError):
    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"'{package}' is required for this backend but is not installed. "
            f"Install it with: pip install 'mlops-toolbox[{extra}]'"
        )


class ValidationFailedError(MLOpsToolboxError):
    """Raised when data validation fails and the caller requested strict mode."""


class TrackingError(MLOpsToolboxError):
    """Raised for experiment tracking failures (e.g. no active run)."""


class RegistryError(MLOpsToolboxError):
    """Raised for model registry failures (e.g. unknown model/version)."""


class MonitoringError(MLOpsToolboxError):
    """Raised for drift-monitoring failures (e.g. no stored reference dataset)."""


class ProjectError(MLOpsToolboxError):
    """Base exception for project-registry failures."""


class ProjectNotFoundError(ProjectError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"No project registered with name {name!r}.")


class ProjectAlreadyRegisteredError(ProjectError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"A project named {name!r} is already registered. Pass overwrite=True to replace it."
        )
