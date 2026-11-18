"""The stores of the runtime layer: where the event log's bytes live and what is read back.

``jsonl`` is the append-only file per conversation tree; ``session`` is the versioned session
projection, its migration and the fold of a log; ``stack`` folds a log into its whole stack of
frames and finds the cut of a fork. This ``__init__`` is the one import surface
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
from .stack import FrameProjection, StackProjection, fold_stack, turn_cut

__all__ = [
    "FrameProjection",
    "JsonlWriter",
    "SessionProjection",
    "SessionVersionError",
    "StackProjection",
    "check_conversation_id",
    "fold_session",
    "fold_stack",
    "migrate_session",
    "projection_from_legacy_file",
    "read_events",
    "turn_cut",
]
