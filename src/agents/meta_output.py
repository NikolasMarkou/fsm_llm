"""
Output formatting and file writing for meta-agent artifacts.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from fsm_llm.logging import logger

from .definitions import MetaBuilderResult
from .exceptions import OutputError


def format_artifact_json(artifact: dict[str, Any]) -> str:
    """Pretty-print an artifact dict as JSON.

    :param artifact: The artifact dict
    :return: Formatted JSON string
    """
    return json.dumps(artifact, indent=2, ensure_ascii=False)


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    """Write an artifact dict to a JSON file.

    The function trusts its caller, like any file-write API: it resolves
    ``path`` (so ``..`` segments are collapsed) and writes there, creating
    parent directories as needed. It does not confine writes to any base
    directory; a caller taking the path from untrusted input must check it.

    :param artifact: The artifact dict
    :param path: Output file path
    :return: Resolved path that was written
    :raises OutputError: If writing fails
    """
    path = Path(path).resolve()

    # Serialise first (a bad artifact creates nothing), then write a temp file
    # beside the target and rename it over: a failed write leaves the old
    # artifact whole. The finally-with-unlink is a no-op after the rename.
    tmp: str | None = None
    try:
        data = format_artifact_json(artifact).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        logger.info(f"Artifact saved to {path}")
        return path
    except (OSError, TypeError, ValueError) as e:
        raise OutputError(
            f"Failed to save artifact: {e}",
            path=str(path),
        ) from e
    finally:
        if tmp is not None:
            with suppress(OSError):
                os.unlink(tmp)


def format_summary(result: MetaBuilderResult) -> str:
    """Format a human-readable summary of a build result.

    :param result: The MetaBuilderResult
    :return: Formatted summary string
    """
    parts: list[str] = []
    parts.append(f"Artifact type: {result.artifact_type.value}")

    name = result.artifact.get("name", "unnamed")
    parts.append(f"Name: {name}")
    parts.append(f"Valid: {'yes' if result.is_valid else 'no'}")
    parts.append(f"Conversation turns: {result.conversation_turns}")

    if result.validation_errors:
        parts.append(f"Validation errors ({len(result.validation_errors)}):")
        for e in result.validation_errors:
            parts.append(f"  - {e}")

    return "\n".join(parts)
