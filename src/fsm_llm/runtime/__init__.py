"""
The runtime layer: the durable event log (plan 21e6f00d, roadmap rows 1.1 to 1.3).

A layer package that ships with the core. It sits between ``core`` and ``llm`` below it and
the L3 siblings above it, so it imports ``core``, ``llm`` and the kernel only, and is never
imported by them. It holds the vocabulary and the safe encoder of the format (``events``),
its flat constants, the blob store (``blobs``) and the JSONL store (``stores``); the log
object lands in a later step.

This ``__init__`` is the one import surface of the package (a static ``__all__``);
a private name is imported from the module that defines it.
"""

from __future__ import annotations

from .blobs import BlobStore, blob_ref
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

__all__ = [
    "DECLARED_TYPES",
    "LOG_FORMAT_VERSION",
    "PRODUCED_TYPES",
    "BlobStore",
    "EventLogError",
    "EventPayloadError",
    "EventType",
    "LogCorruptError",
    "LogLockedError",
    "LogWriteError",
    "TurnOutcome",
    "blob_ref",
    "canonical_json",
    "check_depth",
    "decode_event",
    "encode_event",
    "to_jsonable",
]
