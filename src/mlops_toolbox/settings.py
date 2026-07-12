"""Toolbox-wide configuration: `mlops.toml` plus `MT_*` environment variables.

Resolution precedence (highest wins):

1. Explicit arguments (CLI flags, constructor kwargs)
2. ``MT_*`` environment variables — nested fields use ``__``, e.g. ``MT_AWS__REGION``
3. The nearest ``mlops.toml``, searched upward from the working directory
4. Built-in defaults — identical to the historical local-first behavior

Secrets never belong in ``mlops.toml``: cloud credentials come from each
provider's default chain (boto3 chain, GCP application-default credentials,
``DefaultAzureCredential``) and API keys are read only from environment
variables, so a checked-in config file can never leak them.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydantic
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_FILENAME = "mlops.toml"


def find_config_file(start: Path | None = None) -> Path | None:
    """Nearest `mlops.toml` from `start` (default: cwd) upward, like git discovery."""
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        path = candidate / CONFIG_FILENAME
        if path.is_file():
            return path
    return None


class ServeSettings(pydantic.BaseModel):
    """Defaults for `mt serve` and the shipped serving runtime."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1


class AwsSettings(pydantic.BaseModel):
    """AWS / SageMaker targets. Credentials come from the boto3 default chain."""

    region: str | None = None
    execution_role_arn: str | None = None
    artifact_bucket: str | None = None
    ecr_repository: str | None = None


class GcpSettings(pydantic.BaseModel):
    """GCP / Vertex AI targets. Credentials come from application-default credentials."""

    project: str | None = None
    region: str | None = None
    staging_bucket: str | None = None
    artifact_registry_repo: str | None = None


class AzureSettings(pydantic.BaseModel):
    """Azure ML targets. Credentials come from DefaultAzureCredential."""

    subscription_id: str | None = None
    resource_group: str | None = None
    workspace_name: str | None = None
    acr_name: str | None = None


class _TomlSource(TomlConfigSettingsSource):
    """TOML source tolerating a UTF-8 BOM — Windows editors and PowerShell
    write one by default, and stock tomllib rejects it."""

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        return tomllib.loads(file_path.read_text(encoding="utf-8-sig"))


class ToolboxSettings(BaseSettings):
    """Every knob the toolbox reads, with local-first zero-config defaults."""

    model_config = SettingsConfigDict(
        env_prefix="MT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    tracking_uri: str = "sqlite:///mlops.db"
    registry_uri: str | None = None
    default_deploy_target: str = "local"
    serve: ServeSettings = pydantic.Field(default_factory=ServeSettings)
    aws: AwsSettings = pydantic.Field(default_factory=AwsSettings)
    gcp: GcpSettings = pydantic.Field(default_factory=GcpSettings)
    azure: AzureSettings = pydantic.Field(default_factory=AzureSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _TomlSource(settings_cls, toml_file=find_config_file()),
        )


def load_settings(**overrides: Any) -> ToolboxSettings:
    """Resolve settings fresh on every call — cheap, and honors cwd/env changes."""
    return ToolboxSettings(**overrides)


@dataclass(frozen=True)
class ResolvedField:
    """One resolved setting with where its value came from (for `mt config`)."""

    key: str  # dotted path, e.g. "aws.region"
    value: Any
    source: str  # "env" | "file" | "default"


def explain_settings(settings: ToolboxSettings | None = None) -> list[ResolvedField]:
    """Flatten resolved settings into (key, value, source) rows for display."""
    resolved = settings if settings is not None else load_settings()
    config_path = find_config_file()
    file_data: dict[str, Any] = {}
    if config_path is not None:
        file_data = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))

    rows: list[ResolvedField] = []
    for name in type(resolved).model_fields:
        value = getattr(resolved, name)
        if isinstance(value, pydantic.BaseModel):
            section = file_data.get(name)
            section_data = section if isinstance(section, dict) else {}
            for sub_name in type(value).model_fields:
                rows.append(
                    _resolve_leaf(
                        key=f"{name}.{sub_name}",
                        value=getattr(value, sub_name),
                        env_var=f"MT_{name.upper()}__{sub_name.upper()}",
                        in_file=sub_name in section_data,
                    )
                )
        else:
            rows.append(
                _resolve_leaf(
                    key=name,
                    value=value,
                    env_var=f"MT_{name.upper()}",
                    in_file=name in file_data,
                )
            )
    return rows


def _resolve_leaf(key: str, value: Any, env_var: str, in_file: bool) -> ResolvedField:
    if env_var in os.environ:
        return ResolvedField(key=key, value=value, source="env")
    if in_file:
        return ResolvedField(key=key, value=value, source="file")
    return ResolvedField(key=key, value=value, source="default")
