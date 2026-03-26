from __future__ import annotations

"""
Output formatting and file writing for meta-agent artifacts.
"""

import json
from pathlib import Path
from typing import Any

from fsm_llm.logging import logger

from .definitions import MetaAgentResult
from .exceptions import OutputError


def format_artifact_json(artifact: dict[str, Any]) -> str:
    """Pretty-print an artifact dict as JSON.

    :param artifact: The artifact dict
    :return: Formatted JSON string
    """
    return json.dumps(artifact, indent=2, ensure_ascii=False)


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    """Write an artifact dict to a JSON file.

    :param artifact: The artifact dict
    :param path: Output file path
    :return: Resolved path that was written
    :raises OutputError: If writing fails
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = format_artifact_json(artifact)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Artifact saved to {path}")
        return path.resolve()
    except (OSError, TypeError, ValueError) as e:
        raise OutputError(
            f"Failed to save artifact: {e}",
            path=str(path),
        ) from e


def format_summary(result: MetaAgentResult) -> str:
    """Format a human-readable summary of a build result.

    :param result: The MetaAgentResult
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
