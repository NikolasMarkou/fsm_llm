"""Executables by id (roadmap Phase E2; plan ace7a8c8 D-059): the protocol, the FSM-body adapter,
the registry with template retention, weighted aliases and the two namespaces.

Imported by name (``from fsm_llm.engine.executables import ExecutableRegistry``);
``import fsm_llm`` loads none of it, and ``fsm_llm.engine.api`` and the engine packages below it
import none of it (an import contract). It imports no L6 package: a package declares its
built-ins through the ``executable_sources`` namespace.
"""

from .adapters import FsmBodyExecutable, FsmInstance
from .aliases import AliasSpec, is_template_identity, make_alias, pick_route
from .code_backed import CodeBackedExecutable, CodeBackedInstance
from .namespaces import EXECUTABLE_KINDS, EXECUTABLE_SOURCES, adapter_for
from .protocol import (
    DEFAULT_EXECUTABLE_MAX_STEPS,
    Executable,
    ExecutableCollisionError,
    ExecutableError,
    ExecutableEvictedError,
    ExecutableKindError,
    ExecutableNotFoundError,
    Instance,
)
from .registry import (
    MAX_DOCUMENT_BYTES,
    MAX_RETAINED_TEMPLATES,
    ExecutableRegistry,
    ReloadReport,
    Resolution,
)
from .runs import (
    DEFAULT_MAX_RUNS,
    MAX_FINISHED_RUNS,
    MAX_HANDLE_LENGTH,
    MAX_RESERVED_RUNS,
    ExecutableTargets,
    RunHost,
)

__all__ = [
    "DEFAULT_EXECUTABLE_MAX_STEPS",
    "DEFAULT_MAX_RUNS",
    "EXECUTABLE_KINDS",
    "EXECUTABLE_SOURCES",
    "MAX_DOCUMENT_BYTES",
    "MAX_FINISHED_RUNS",
    "MAX_HANDLE_LENGTH",
    "MAX_RESERVED_RUNS",
    "MAX_RETAINED_TEMPLATES",
    "AliasSpec",
    "CodeBackedExecutable",
    "CodeBackedInstance",
    "Executable",
    "ExecutableCollisionError",
    "ExecutableError",
    "ExecutableEvictedError",
    "ExecutableKindError",
    "ExecutableNotFoundError",
    "ExecutableRegistry",
    "ExecutableTargets",
    "FsmBodyExecutable",
    "FsmInstance",
    "Instance",
    "ReloadReport",
    "Resolution",
    "RunHost",
    "adapter_for",
    "is_template_identity",
    "make_alias",
    "pick_route",
]
