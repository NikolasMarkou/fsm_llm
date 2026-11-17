"""The stores of the runtime layer: where the event log's bytes live.

``jsonl`` is the append-only file per conversation tree. This ``__init__`` is the one import
surface of the package (a static ``__all__``); a private name is imported from its module.
"""

from __future__ import annotations

from .jsonl import JsonlWriter, check_conversation_id, read_events

__all__ = ["JsonlWriter", "check_conversation_id", "read_events"]
