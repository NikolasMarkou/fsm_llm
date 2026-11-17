"""
The runtime layer: the durable event log (plan 21e6f00d, roadmap rows 1.1 to 1.3).

A layer package that ships with the core. It sits between ``core`` and ``llm`` below it and
the L3 siblings above it, so it imports ``core``, ``llm`` and the kernel only, and is never
imported by them. This step holds the vocabulary and the safe encoder of the format
(``events``) and its flat constants; the writer, the blob store and the log object land in
the next steps.

This ``__init__`` is the one import surface of the package (a static ``__all__``);
a private name is imported from the module that defines it.
"""

from __future__ import annotations

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
    check_depth,
    decode_event,
    encode_event,
    to_jsonable,
)

__all__ = [
    "DECLARED_TYPES",
    "LOG_FORMAT_VERSION",
    "PRODUCED_TYPES",
    "EventLogError",
    "EventPayloadError",
    "EventType",
    "LogCorruptError",
    "LogLockedError",
    "LogWriteError",
    "TurnOutcome",
    "check_depth",
    "decode_event",
    "encode_event",
    "to_jsonable",
]
