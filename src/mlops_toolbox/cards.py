"""Model cards rendered from what the registry already knows about a model."""

from __future__ import annotations

from mlops_toolbox.core.models import ModelContract, ModelInfo
from mlops_toolbox.registry.base import ModelRegistry

__all__ = ["generate_model_card"]

_HAS_REFERENCE_TAG = "mlops_toolbox.has_reference"


def generate_model_card(registry: ModelRegistry, name: str, version: str = "latest") -> str:
    """Render a markdown model card: identity, input schema, output, metrics, monitoring."""
    info = registry.get_model_info(name, version)
    contract = registry.get_contract(name, version)

    sections = [_header(info), _schema_section(contract), _evaluation_section(contract)]
    sections.append(_monitoring_section(info))
    return "\n\n".join(section for section in sections if section) + "\n"


def _header(info: ModelInfo) -> str:
    lines = [
        f"# Model card: {info.name}",
        "",
        "| | |",
        "|---|---|",
        f"| Version | {info.version} |",
        f"| Aliases | {', '.join(info.aliases) if info.aliases else '—'} |",
        f"| URI | `{info.uri}` |",
        f"| Source run | `{info.source_run_id or '—'}` |",
    ]
    user_tags = {k: v for k, v in info.tags.items() if not k.startswith("mlops_toolbox.")}
    if user_tags:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(user_tags.items()))
        lines.append(f"| Tags | {rendered} |")
    return "\n".join(lines)


def _schema_section(contract: ModelContract | None) -> str:
    if contract is None:
        return (
            "## Input schema\n\nNo contract stored for this version — register with "
            "`contract=mt.infer_contract(X_train, ...)` to capture one."
        )
    lines = [
        "## Input schema",
        "",
        "| Column | Type | Constraints |",
        "|---|---|---|",
    ]
    for spec in contract.input_columns:
        if spec.categories:
            constraint = "one of: " + ", ".join(f"`{c}`" for c in spec.categories)
        elif spec.min_value is not None and spec.max_value is not None:
            constraint = f"observed range {spec.min_value:g} – {spec.max_value:g}"
        else:
            constraint = "—"
        lines.append(f"| `{spec.name}` | {spec.dtype} | {constraint} |")
    if contract.output is not None:
        lines += ["", f"**Output**: {contract.output.dtype}"]
        if contract.output.labels:
            lines[-1] += " — labels: " + ", ".join(f"`{v}`" for v in contract.output.labels)
    return "\n".join(lines)


def _evaluation_section(contract: ModelContract | None) -> str:
    if contract is None or contract.evaluation is None:
        return ""
    report = contract.evaluation
    lines = [
        "## Evaluation",
        "",
        f"Task: {report.task_type} · evaluated on {report.n_samples} samples",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {metric} | {value:.6g} |" for metric, value in sorted(report.metrics.items())]
    return "\n".join(lines)


def _monitoring_section(info: ModelInfo) -> str:
    if info.tags.get(_HAS_REFERENCE_TAG) == "true":
        return (
            "## Monitoring\n\nDrift baseline stored at registration — check with "
            f"`mt drift {info.name} --input <current-data>`."
        )
    return (
        "## Monitoring\n\nNo drift baseline stored — register with "
        "`reference_data=X_train` to enable drift checks."
    )
