"""
The runtime layer: the durable event log (plan 21e6f00d, roadmap rows 1.1 to 1.3).

A layer package that ships with the core. It sits between ``core`` and ``llm`` below it and
the L3 siblings above it, so it imports ``core``, ``llm`` and the kernel only, and is never
imported by them. It holds the vocabulary and the safe encoder of the format (``events``),
its flat constants, the blob store (``blobs``), the JSONL store (``stores``), the checkpoint
rule (``checkpoint``), the log object that ties them together (``log.EventLog``) and the turn
scope the engine reaches it through (``scope``).

This ``__init__`` is the one import surface of the package (a static ``__all__``);
a private name is imported from the module that defines it.
"""

from __future__ import annotations

from .blobs import BlobStore, blob_ref
from .checkpoint import CHECKPOINT_TYPES, is_checkpoint, unresolved_tool_calls
from .constants import LOG_FORMAT_VERSION
from .events import (
    DECLARED_TYPES,
    PRODUCED_TYPES,
    EventLogError,
    EventPayloadError,
    EventType,
    LogCorruptError,
    LogLockedError,
    LogWriteError,
    TurnOutcome,
    canonical_json,
    check_depth,
    decode_event,
    encode_event,
    to_jsonable,
)
from .log import EventLog, filter_context, warn_on_failure
from .scope import (
    LlmCall,
    TurnScope,
    activate,
    bind,
    current_scope,
    emit,
    llm_call,
    scoped_iter,
)

__all__ = [
    "CHECKPOINT_TYPES",
    "DECLARED_TYPES",
    "LOG_FORMAT_VERSION",
    "PRODUCED_TYPES",
    "BlobStore",
    "EventLog",
    "EventLogError",
    "EventPayloadError",
    "EventType",
    "LlmCall",
    "LogCorruptError",
    "LogLockedError",
    "LogWriteError",
    "TurnOutcome",
    "TurnScope",
    "activate",
    "bind",
    "blob_ref",
    "canonical_json",
    "check_depth",
    "current_scope",
    "decode_event",
    "emit",
    "encode_event",
    "filter_context",
    "is_checkpoint",
    "llm_call",
    "scoped_iter",
    "to_jsonable",
    "unresolved_tool_calls",
    "warn_on_failure",
]
