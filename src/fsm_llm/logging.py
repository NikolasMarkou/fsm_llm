import os
import sys
from loguru import logger
from functools import wraps
from typing import Union, Callable

# --------------------------------------------------------------


# Define a filter to handle logs without conversation_id
def prepare_log_record(record):
    if "conversation_id" not in record["extra"]:
        record["extra"]["conversation_id"] = "GENERAL"
    return record


# Remove default handler
logger.remove()

# Track handler IDs added by this library (for safe removal in enable_debug_logging)
_library_handler_ids = []

# --------------------------------------------------------------


# Add console handler with colors and conversation_id
_library_handler_ids.append(logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<yellow>conv_id: {extra[conversation_id]:<12}</yellow> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    level="INFO",
    filter=prepare_log_record
))

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
        method_or_error_msg: Union[Callable, str] = None):
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
                logger.error(f"Error in {method.__name__}: {str(e)}")
                raise FSMError(f"{error_message}: {str(e)}") from e
        return wrapper

    # Handle both @handle_conversation_errors and @handle_conversation_errors("msg")
    if callable(method_or_error_msg):
        return decorator(method_or_error_msg)
    return decorator

# --------------------------------------------------------------
