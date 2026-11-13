"""
The tool layer: the tool models, the registry that runs them and the helpers around both.

A layer package: ``models`` (``ToolDefinition``, ``ToolCall``, ``ToolResult``,
``ToolAnnotations``), ``registry`` (``ToolRegistry``), ``decorator`` (``@tool``),
``schema`` (the JSON parameter schema inferred from type hints), ``arguments`` (what
a model sent turned into call arguments, and ``redact_secret_entries``), ``hints``,
``timeouts`` (the per-call timeout), ``errors`` (``ToolExecutionError``,
``ToolNotFoundError``), ``limits`` (``require_int_limit``, ``bounded_repr``),
``truncation`` and ``constants`` (the subset of the agents constants these modules
read, plus ``ToolRunStatus`` and ``ToolObservationPrefix``). The agents package builds
on it (``fsm_llm.agents`` re-exports the public names); the tools never import
``agents``.

Imports the kernel (``logging``), ``core`` (``AgentError``) and one helper of the
``context`` layer (``context.filtering.redact_context``, the one named debt of the
contract) and nothing above.

This ``__init__`` is the one import surface of the package (a static ``__all__``);
a private name is imported from the module that defines it.
"""

from __future__ import annotations

from .arguments import redact_secret_entries
from .constants import ToolObservationPrefix, ToolRunStatus
from .decorator import tool
from .errors import ToolExecutionError, ToolNotFoundError
from .hints import callable_name
from .limits import bounded_repr, require_int_limit
from .models import (
    READ_ONLY_ANNOTATIONS,
    ToolAnnotations,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from .registry import ToolRegistry, registry_from
from .schema import infer_schema_from_hints, schema_types, tool_parameters_schema
from .timeouts import ToolTimedOut, call_with_timeout, check_timeout
from .truncation import smart_truncate

__all__ = [
    "READ_ONLY_ANNOTATIONS",
    "ToolAnnotations",
    "ToolCall",
    "ToolDefinition",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolObservationPrefix",
    "ToolRegistry",
    "ToolResult",
    "ToolRunStatus",
    "ToolTimedOut",
    "bounded_repr",
    "call_with_timeout",
    "callable_name",
    "check_timeout",
    "infer_schema_from_hints",
    "redact_secret_entries",
    "registry_from",
    "require_int_limit",
    "schema_types",
    "smart_truncate",
    "tool",
    "tool_parameters_schema",
]
