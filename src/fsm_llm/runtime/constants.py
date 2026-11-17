"""Flat constants of the runtime layer: the event log format, its caps and its names.

Naming follows the repository rule: ``MAX_*`` is a hard cap, ``*_BYTES`` carries its unit,
``*_DIR`` is a directory name under a log root, ``*_RE`` is a compiled pattern matched with
``fullmatch`` (never ``match`` or ``$``, which accepts a trailing newline).
"""

from __future__ import annotations

import re

#: The ``v`` of every event this package writes. A reader refuses a larger one.
LOG_FORMAT_VERSION = 1

#: Hard cap of one encoded event line, newline included. A larger payload is a blob.
MAX_EVENT_BYTES = 65_536

#: Hard cap of one blob (16 MiB); a larger payload is refused, never truncated.
MAX_BLOB_BYTES = 16 * 1024 * 1024

#: Hard cap of the nesting depth of any payload, checked iteratively on write and on read.
#: The envelope itself is depth 1.
MAX_PAYLOAD_DEPTH = 32

#: Directory names under a log root: ``<root>/logs/<id>.jsonl`` and ``<root>/blobs/<aa>/<62 hex>``.
LOG_DIR = "logs"
BLOB_DIR = "blobs"

#: The keys every event carries, in the roadmap's order. The rest is per event type.
ENVELOPE_KEYS = ("v", "seq", "ts", "conversation_id", "frame", "type")

#: A conversation id usable as a file name on every platform: lowercase only, so two ids never
#: differ only by case on a case-folding file system. A uuid4 passes.
CONVERSATION_ID_RE = re.compile(r"[a-z0-9_-]{1,64}")

#: Windows device names, reserved with or without an extension and in any case.
WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(1, 10)}
    | {f"lpt{n}" for n in range(1, 10)}
)

#: A content address: ``sha256:`` and 64 lowercase hex digits.
BLOB_REF_RE = re.compile(r"sha256:[0-9a-f]{64}")

#: A payload whose canonical JSON is at most this many bytes is stored inline in its event;
#: a larger one is a blob and the event keeps only its reference (an event is capped at
#: ``MAX_EVENT_BYTES``, so the inline limit leaves room for the rest of the event).
MAX_INLINE_BYTES = 8_192
