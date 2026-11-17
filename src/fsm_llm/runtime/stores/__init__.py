"""The stores of the runtime layer: where the event log's bytes live and what is read back.

``jsonl`` is the append-only file per conversation tree; ``session`` is the versioned session
projection, its migration and the fold of a log. This ``__init__`` is the one import surface
of the package (a static ``__all__``); a private name is imported from its module.
"""

from __future__ import annotations

from .jsonl import JsonlWriter, check_conversation_id, read_events
from .session import (
    SessionProjection,
    SessionVersionError,
    fold_session,
    migrate_session,
    projection_from_legacy_file,
)

__all__ = [
    "JsonlWriter",
    "SessionProjection",
    "SessionVersionError",
    "check_conversation_id",
    "fold_session",
    "migrate_session",
    "projection_from_legacy_file",
    "read_events",
]
