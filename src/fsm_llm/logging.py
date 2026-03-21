from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps

from loguru import logger

# --------------------------------------------------------------


# Define a filter to handle logs without conversation_id
def prepare_log_record(record):
    if "conversation_id" not in record["extra"]:
        record["extra"]["conversation_id"] = "GENERAL"
    return record


# Track handler IDs added by this library (for safe removal in enable_debug_logging)
_library_handler_ids = []

# --------------------------------------------------------------


# Do NOT add handlers on import — libraries should not configure logging.
# Users who want console logging can call enable_debug_logging() or add
# their own loguru handler.  We only disable the default loguru handler
# so that unconfigured usage does not spam stderr.
logger.disable("fsm_llm")

# --------------------------------------------------------------

_file_handler_initialized = False


def setup_file_logging(log_dir="logs"):
    """Set up file logging. Call this explicitly when file logging is needed."""
    global _file_handler_initialized
    if _file_handler_initialized:
        return
    _file_handler_initialized = True

    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, "fsm-llm_{time}.log"),
        rotation="10 MB",
        retention="1 month",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | "
               "{level: <8} | "
               "conv_id: {extra[conversation_id]:<12} | "
               "{name}:{function}:{line} | "
               "{message}",
        level="DEBUG",
        filter=prepare_log_record
    )

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
        method_or_error_msg: Callable | str = None):
    """
    Decorator for handling common conversation-related errors in FSM_LLM methods.

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
