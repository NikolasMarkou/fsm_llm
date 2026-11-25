"""The workflow presets of the dashboard (plan 7d9c0955 step 7; the old ``monitor/instance_manager.py``, same dotted path).

A preset is a ``workflow`` document of the package directory ``monitor/workflows/`` and nothing
else; the named callables its steps use are the monitor's own (``preset_callables()``). Presets use
only pure steps (auto and condition), so they run with no model and no API.

Interface contract: ``load_preset_registry() -> ExecutableRegistry`` reads the directory and raises
``RuntimeError`` (it assigns nothing) when a document is refused or the directory serves no
workflow, and the ``WorkflowValidationError`` of a document that does not build against
``preset_callables()``; ``build_preset(registry, preset_id)`` returns a FRESH ``WorkflowDefinition`` per
call (the registry's ``ExecutableNotFoundError`` for an id it lacks). Imports no other module of the
package.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, cast

from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.workflows import WorkflowDefinition
from fsm_llm.workflows.constants import CALLABLES_NAMESPACE
from fsm_llm.workflows.executable import WorkflowExecutable  # registers the kind


def ingest(context: dict[str, Any]) -> dict[str, Any]:
    return {"ingested": True}


def process(context: dict[str, Any]) -> dict[str, Any]:
    return {"processed": True}


def amount_is_high(context: dict[str, Any]) -> bool:
    return float(context.get("amount", 0) or 0) >= 1000


def mark_high(context: dict[str, Any]) -> dict[str, Any]:
    return {"tier": "high"}


def mark_low(context: dict[str, Any]) -> dict[str, Any]:
    return {"tier": "low"}


def preset_callables() -> NameRegistry[Any]:
    """The five functions a preset document names, in a registry built per call (never shared)."""
    registry: NameRegistry[Any] = NameRegistry(CALLABLES_NAMESPACE)
    for function in (ingest, process, amount_is_high, mark_high, mark_low):
        registry.register(function.__name__, function)
    return registry


# DECISION plan-2026-11-25T195353-7d9c0955/D-011
# The dashboard offers only the workflow documents of its OWN package directory, built from the
# monitor's named callables; a request never names a file, a callable or a step. A load that fails
# raises and the caller assigns nothing, so the next call retries. Do NOT accept a document or a
# path from a request, serve a user directory as presets, or load this on the loop thread, in
# `__init__` or under `_lock` (the caller holds the leaf `_catalog_lock`).
def load_preset_registry() -> ExecutableRegistry:
    """A private registry over ``monitor/workflows/`` (see the module docstring)."""
    directory = Path(str(resources.files("fsm_llm.monitor").joinpath("workflows")))
    registry = ExecutableRegistry([directory])
    report = registry.reload()
    if report.failed or report.collisions or not registry.list(kind="workflow"):
        raise RuntimeError(
            f"the workflow presets of {directory.name!r} did not load: "
            f"{sorted(report.failed) or sorted(report.collisions) or 'no workflow'}"
        )
    for executable in registry.list(kind="workflow"):
        build_preset(registry, executable.id)
    return registry


def build_preset(registry: ExecutableRegistry, preset_id: str) -> WorkflowDefinition:
    """A fresh definition of one preset, so concurrent launches never share instance state."""
    executable = cast(WorkflowExecutable, registry.get(preset_id))
    return executable.build_definition(preset_callables())
