from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

from loguru import logger

from .constants import (
    ENV_LOG_FORMAT,
    ENV_LOG_LEVEL,
    LOG_DEFAULT_COMPRESSION,
    LOG_DEFAULT_CONVERSATION_ID,
    LOG_DEFAULT_FILE_PATTERN,
    LOG_DEFAULT_LEVEL,
    LOG_DEFAULT_RETENTION,
    LOG_DEFAULT_ROTATION,
    LOG_FILE_FORMAT,
    LOG_FORMAT_HUMAN,
    LOG_FORMAT_JSON,
    LOG_HUMAN_FORMAT,
    LOG_HUMAN_FORMAT_WITH_CONTEXT,
    LOG_SINK_FILE,
    LOG_SINK_STDERR,
    LOG_SINK_STDOUT,
)

# --------------------------------------------------------------


def prepare_log_record(record):
    """Filter to ensure all log records have standard context fields."""
    if "conversation_id" not in record["extra"]:
        record["extra"]["conversation_id"] = LOG_DEFAULT_CONVERSATION_ID
    if "package" not in record["extra"]:
        record["extra"]["package"] = "fsm_llm"
    return record


# Track handler IDs added by this library (for safe removal)
_library_handler_ids: list[int] = []

# --------------------------------------------------------------


# Do NOT add handlers on import — libraries should not configure logging.
# Users who want console logging can call enable_debug_logging() or add
# their own loguru handler.  We only disable the default loguru handler
# so that unconfigured usage does not spam stderr.
logger.disable("fsm_llm")

# --------------------------------------------------------------

_file_handler_initialized = False


def _format_json(record) -> str:
    """Format a log record as a single-line JSON string (JSONL).

    Produces flat JSON compatible with Grafana Loki, ELK, Datadog,
    CloudWatch, and other log aggregation systems.
    """
    extra = record["extra"]
    entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    # Include all extra fields (conversation_id, package, agent_type, etc.)
    for key, value in extra.items():
        entry[key] = value
    # Include exception info if present
    if record["exception"] is not None:
        entry["exception_type"] = record["exception"].type.__name__
        entry["exception_message"] = str(record["exception"].value)
    return json.dumps(entry, default=str) + "\n"


def setup_logging(
    *,
    sink: str = LOG_SINK_STDERR,
    format: str | None = None,
    level: str | None = None,
    log_dir: str = "logs",
    rotation: str = LOG_DEFAULT_ROTATION,
    retention: str = LOG_DEFAULT_RETENTION,
    compression: str = LOG_DEFAULT_COMPRESSION,
    context: bool = False,
) -> int:
    """Configure logging for fsm_llm.

    Unified configuration entry point supporting streaming (stderr/stdout),
    rotating files, and structured JSON output for log aggregation systems.

    Args:
        sink: Output destination — "stderr", "stdout", or "file".
        format: Log format — "human" (colored text) or "json" (JSONL).
            If None, reads from FSM_LLM_LOG_FORMAT env var, defaulting to "human".
        level: Log level — "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
            If None, reads from FSM_LLM_LOG_LEVEL env var, defaulting to "DEBUG".
        log_dir: Directory for log files (only used when sink="file").
        rotation: File rotation trigger (e.g., "10 MB", "00:00", "1 week").
        retention: How long to keep old log files (e.g., "1 month", "10 days").
        compression: Compression for rotated files (e.g., "zip", "gz").
        context: If True, include conversation_id in human-readable format.
            Always included in JSON format.

    Returns:
        Handler ID (can be used with logger.remove() to unregister).

    Example:
        # Human-readable to stderr (development)
        setup_logging()

        # JSON to stdout (Docker/K8s log collection)
        setup_logging(sink="stdout", format="json")

        # JSON to rotating files (production)
        setup_logging(sink="file", format="json", level="INFO")

        # Via environment variables
        # FSM_LLM_LOG_LEVEL=WARNING FSM_LLM_LOG_FORMAT=json
        setup_logging()
    """
    # Resolve level from env var or default
    resolved_level = level or os.environ.get(ENV_LOG_LEVEL, LOG_DEFAULT_LEVEL)

    # Resolve format from env var or default
    resolved_format = format or os.environ.get(ENV_LOG_FORMAT, LOG_FORMAT_HUMAN)

    # Enable library logging
    logger.enable("fsm_llm")

    # Determine sink and format
    if sink == LOG_SINK_FILE:
        global _file_handler_initialized
        if _file_handler_initialized:
            return -1
        _file_handler_initialized = True
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, LOG_DEFAULT_FILE_PATTERN)

        if resolved_format == LOG_FORMAT_JSON:
            handler_id = logger.add(
                file_path,
                format=_format_json,
                rotation=rotation,
                retention=retention,
                compression=compression,
                level=resolved_level,
                filter=prepare_log_record,
            )
        else:
            handler_id = logger.add(
                file_path,
                format=LOG_FILE_FORMAT,
                rotation=rotation,
                retention=retention,
                compression=compression,
                level=resolved_level,
                filter=prepare_log_record,
            )
    else:
        output = sys.stdout if sink == LOG_SINK_STDOUT else sys.stderr

        if resolved_format == LOG_FORMAT_JSON:
            handler_id = logger.add(
                output,
                format=_format_json,
                level=resolved_level,
                filter=prepare_log_record,
                colorize=False,
            )
        else:
            fmt = LOG_HUMAN_FORMAT_WITH_CONTEXT if context else LOG_HUMAN_FORMAT
            handler_id = logger.add(
                output,
                format=fmt,
                level=resolved_level,
                filter=prepare_log_record,
                colorize=True,
            )

    _library_handler_ids.append(handler_id)
    return handler_id


# --------------------------------------------------------------


def setup_file_logging(log_dir="logs"):
    """Set up file logging with rotation. Call explicitly when file logging is needed.

    This is a convenience wrapper around setup_logging(sink="file").
    """
    global _file_handler_initialized
    if _file_handler_initialized:
        return
    _file_handler_initialized = True

    os.makedirs(log_dir, exist_ok=True)
    handler_id = logger.add(
        os.path.join(log_dir, LOG_DEFAULT_FILE_PATTERN),
        rotation=LOG_DEFAULT_ROTATION,
        retention=LOG_DEFAULT_RETENTION,
        compression=LOG_DEFAULT_COMPRESSION,
        format=LOG_FILE_FORMAT,
        level=LOG_DEFAULT_LEVEL,
        filter=prepare_log_record,
    )
    _library_handler_ids.append(handler_id)


# --------------------------------------------------------------


def with_conversation_context(func):
    @wraps(func)  # Preserve function metadata
    def wrapper(self, conversation_id, *args, **kwargs):
        # Bind the conversation ID to the logger
        log = logger.bind(conversation_id=conversation_id)

        # Call the original function with the contextual logger
        return func(self, conversation_id, *args, log=log, **kwargs)

    return wrapper

# --------------------------------------------------------------


def handle_conversation_errors(
        method_or_error_msg: Callable[..., Any] | str | None = None):
    """Decorator for handling common conversation-related errors in FSM_LLM methods.

    Can be used in two ways:
    1. @handle_conversation_errors - uses the method name in error messages
    2. @handle_conversation_errors("Custom error message") - uses the provided message
    """
    def decorator(method):
        from .definitions import FSMError

        @wraps(method)
        def wrapper(self, conversation_id, *args, **kwargs):
            error_message = (
                method_or_error_msg if isinstance(method_or_error_msg, str)
                else f"Failed in {method.__name__}"
            )

            try:
                return method(self, conversation_id, *args, **kwargs)
            except (ValueError, FSMError):
                raise
            except Exception as e:
                logger.error(f"Error in {method.__name__}: {e!s}")
                raise FSMError(f"{error_message}: {e!s}") from e
        return wrapper

    # Handle both @handle_conversation_errors and @handle_conversation_errors("msg")
    if callable(method_or_error_msg):
        return decorator(method_or_error_msg)
    return decorator

# --------------------------------------------------------------
