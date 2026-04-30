from __future__ import annotations

"""fsm_llm.debug — development and debug helpers.

Moved from the top-level ``fsm_llm`` namespace at 0.9.0 (`enable_debug_logging`
and `disable_warnings` were previously importable from `fsm_llm` directly).
``BUFFER_METADATA`` (the working-memory buffer schema constant) is also
re-exported here.

    from fsm_llm.debug import enable_debug_logging, disable_warnings, BUFFER_METADATA
"""

import sys
import warnings

from .memory import BUFFER_METADATA


def enable_debug_logging() -> None:
    """Enable debug logging for development."""
    from .logging import _library_handler_ids, logger, prepare_log_record

    # Re-enable the library loggers
    logger.enable("fsm_llm")

    # Only remove library-registered handlers (not user's handlers)
    for handler_id in _library_handler_ids:
        try:
            logger.remove(handler_id)
        except ValueError:
            pass
    _library_handler_ids.clear()

    # Reset file handler flag so setup_file_logging can be called again
    from . import logging as log_module

    log_module._file_handler_initialized = False

    _library_handler_ids.append(
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
            filter=prepare_log_record,
        )
    )


def disable_warnings() -> None:
    """Disable framework warnings."""
    warnings.filterwarnings("ignore", category=UserWarning, module=r"fsm_llm")


__all__ = [
    "enable_debug_logging",
    "disable_warnings",
    "BUFFER_METADATA",
]
