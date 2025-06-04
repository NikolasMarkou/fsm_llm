# /logging.py

"""
Enhanced logging system for 2-Pass LLM-FSM Architecture.

This module provides comprehensive logging capabilities for the enhanced framework,
including specialized logging for content generation, transition evaluation,
and other 2-pass architecture components.

Key Features:
- Structured logging with conversation context
- Component-specific log levels and formatting
- Performance logging for transition evaluation
- Debug logging for prompt generation
- Error tracking and analysis
- Conversation flow tracing
"""

import sys
import json
import time
from functools import wraps
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from pathlib import Path
from loguru import logger

# --------------------------------------------------------------
# Enhanced Log Configuration
# --------------------------------------------------------------

# Create logs directory structure
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create subdirectories for different log types
(log_dir / "conversations").mkdir(exist_ok=True)
(log_dir / "transitions").mkdir(exist_ok=True)
(log_dir / "performance").mkdir(exist_ok=True)
(log_dir / "errors").mkdir(exist_ok=True)


def prepare_log_record(record):
    """Enhanced log record preparation with additional context."""
    extra = record["extra"]

    # Set default conversation ID if missing
    if "conversation_id" not in extra:
        extra["conversation_id"] = "GENERAL"

    # Add component context if missing
    if "component" not in extra:
        module_name = record.get("name", "").split(".")[-1]
        extra["component"] = module_name.upper() if module_name else "UNKNOWN"

    # Add performance context if missing
    if "performance" not in extra:
        extra["performance"] = False

    # Add transition context if missing
    if "transition_context" not in extra:
        extra["transition_context"] = {}

    return record


# Remove default handler
logger.remove()

# --------------------------------------------------------------
# Console Handler with Enhanced Format
# --------------------------------------------------------------

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<yellow>conv:{extra[conversation_id]:<12}</yellow> | "
           "<blue>{extra[component]:<12}</blue> | "
           "<cyan>{name}:{function}:{line}</cyan> | "
           "<level>{message}</level>",
    level="INFO",
    filter=prepare_log_record,
    colorize=True,
    backtrace=True,
    diagnose=True
)

# --------------------------------------------------------------
# Main Application Log File
# --------------------------------------------------------------

logger.add(
    log_dir / "llm-fsm_{time}.log",
    rotation="10 MB",
    retention="1 month",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
           "{level: <8} | "
           "conv:{extra[conversation_id]:<12} | "
           "{extra[component]:<12} | "
           "{name}:{function}:{line} | "
           "{message}",
    level="DEBUG",
    filter=prepare_log_record,
    serialize=False
)

# --------------------------------------------------------------
# Specialized Log Files
# --------------------------------------------------------------

# Conversation-specific logging
logger.add(
    log_dir / "conversations" / "conversations_{time}.log",
    rotation="5 MB",
    retention="2 weeks",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {extra[conversation_id]} | {level} | {message}",
    filter=lambda record: record["extra"].get("conversation_id") != "GENERAL",
    level="INFO"
)

# Transition evaluation logging
logger.add(
    log_dir / "transitions" / "transitions_{time}.log",
    rotation="5 MB",
    retention="1 week",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {extra[conversation_id]} | {level} | {extra[component]} | {message}",
    filter=lambda record: "transition" in record["extra"].get("component", "").lower(),
    level="DEBUG"
)

# Performance logging
logger.add(
    log_dir / "performance" / "performance_{time}.log",
    rotation="5 MB",
    retention="1 week",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    filter=lambda record: record["extra"].get("performance", False),
    level="INFO"
)

# Error logging
logger.add(
    log_dir / "errors" / "errors_{time}.log",
    rotation="5 MB",
    retention="1 month",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[conversation_id]} | {extra[component]} | {name}:{function}:{line} | {message}",
    filter=lambda record: record["levelno"] >= 40,  # ERROR and CRITICAL
    level="ERROR"
)


# --------------------------------------------------------------
# Context Managers and Decorators
# --------------------------------------------------------------

def with_conversation_context(func):
    """Enhanced decorator for conversation-specific logging context."""

    @wraps(func)
    def wrapper(self, conversation_id, *args, **kwargs):
        # Create enhanced logger binding
        log = logger.bind(
            conversation_id=conversation_id,
            component=self.__class__.__name__.upper(),
            method=func.__name__
        )

        # Add performance tracking
        start_time = time.time()

        try:
            # Call the original function with enhanced logger
            result = func(self, conversation_id, *args, log=log, **kwargs)

            # Log performance if enabled
            duration = time.time() - start_time
            if duration > 1.0:  # Log slow operations
                perf_log = logger.bind(performance=True)
                perf_log.info(f"{self.__class__.__name__}.{func.__name__} took {duration:.2f}s")

            return result

        except Exception as e:
            # Enhanced error logging
            error_log = logger.bind(
                conversation_id=conversation_id,
                component=self.__class__.__name__.upper(),
                error_type=e.__class__.__name__
            )
            error_log.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def with_transition_context(func):
    """Decorator for transition evaluation logging context."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract context information
        current_state = kwargs.get('current_state') or (args[0].id if args else 'unknown')

        log = logger.bind(
            component="TRANSITION_EVALUATOR",
            current_state=current_state,
            transition_context={
                "state": current_state,
                "method": func.__name__
            }
        )

        start_time = time.time()

        try:
            result = func(self, *args, **kwargs)

            # Log transition evaluation performance
            duration = time.time() - start_time
            if duration > 0.1:  # Log evaluations taking > 100ms
                perf_log = logger.bind(performance=True)
                perf_log.info(f"Transition evaluation for {current_state} took {duration:.3f}s")

            return result

        except Exception as e:
            log.error(f"Transition evaluation error in {current_state}: {str(e)}")
            raise

    return wrapper


def with_llm_context(request_type: str):
    """Decorator for LLM request logging context."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, request, *args, **kwargs):
            log = logger.bind(
                component="LLM_INTERFACE",
                request_type=request_type,
                model=getattr(self, 'model', 'unknown')
            )

            start_time = time.time()

            try:
                # Log request details (truncated for privacy)
                if hasattr(request, 'user_message'):
                    message_preview = request.user_message[:100] + "..." if len(
                        request.user_message) > 100 else request.user_message
                    log.debug(f"{request_type} request: {message_preview}")

                result = func(self, request, *args, **kwargs)

                # Log performance
                duration = time.time() - start_time
                perf_log = logger.bind(performance=True)
                perf_log.info(f"LLM {request_type} request completed in {duration:.2f}s")

                return result

            except Exception as e:
                log.error(f"LLM {request_type} request failed: {str(e)}")
                raise

        return wrapper

    return decorator


def with_handler_context(func):
    """Decorator for handler execution logging context."""

    @wraps(func)
    def wrapper(self, timing, *args, **kwargs):
        log = logger.bind(
            component="HANDLER_SYSTEM",
            timing=timing.name if hasattr(timing, 'name') else str(timing),
            handler_count=len(getattr(self, 'handlers', []))
        )

        start_time = time.time()

        try:
            result = func(self, timing, *args, **kwargs)

            # Log handler execution performance
            duration = time.time() - start_time
            if duration > 0.5:  # Log slow handler executions
                perf_log = logger.bind(performance=True)
                perf_log.info(f"Handler execution for {timing} took {duration:.3f}s")

            return result

        except Exception as e:
            log.error(f"Handler execution error at {timing}: {str(e)}")
            raise

    return wrapper


# --------------------------------------------------------------
# Error Handling Decorators
# --------------------------------------------------------------

def handle_conversation_errors(method_or_error_msg: Union[Callable, str] = None):
    """Enhanced decorator for handling conversation-related errors."""

    def decorator(method):
        from .definitions import FSMError

        @wraps(method)
        def wrapper(self, conversation_id, *args, **kwargs):
            error_message = (
                method_or_error_msg if isinstance(method_or_error_msg, str)
                else f"Failed in {method.__name__}"
            )

            # Enhanced logging context
            log = logger.bind(
                conversation_id=conversation_id,
                component=self.__class__.__name__.upper(),
                method=method.__name__
            )

            try:
                return method(self, conversation_id, *args, **kwargs)

            except ValueError as e:
                log.error(f"Invalid conversation ID: {conversation_id}")
                raise ValueError(f"Conversation not found: {conversation_id}")

            except FSMError as e:
                log.error(f"FSM error in {method.__name__}: {str(e)}")
                raise

            except Exception as e:
                log.error(f"Unexpected error in {method.__name__}: {str(e)}")
                raise FSMError(f"{error_message}: {str(e)}")

        return wrapper

    # Handle both @handle_conversation_errors and @handle_conversation_errors("msg")
    if callable(method_or_error_msg):
        return decorator(method_or_error_msg)
    return decorator


# --------------------------------------------------------------
# Specialized Logging Functions
# --------------------------------------------------------------

def log_content_generation(conversation_id: str, state: str, message_length: int, extracted_keys: List[str]):
    """Log content generation details."""
    content_log = logger.bind(
        conversation_id=conversation_id,
        component="CONTENT_GENERATION"
    )

    content_log.info(
        f"Generated content in state '{state}': "
        f"message_length={message_length}, "
        f"extracted_keys={extracted_keys}"
    )


def log_transition_evaluation(
        conversation_id: str,
        current_state: str,
        evaluation_result: str,
        confidence: float,
        transition_count: int,
        evaluation_time: float
):
    """Log transition evaluation results."""
    transition_log = logger.bind(
        conversation_id=conversation_id,
        component="TRANSITION_EVALUATOR",
        transition_context={
            "current_state": current_state,
            "result": evaluation_result,
            "confidence": confidence,
            "options": transition_count
        }
    )

    transition_log.info(
        f"Transition evaluation: {current_state} -> {evaluation_result} "
        f"(confidence={confidence:.2f}, options={transition_count}, time={evaluation_time:.3f}s)"
    )


def log_state_transition(conversation_id: str, from_state: str, to_state: str, method: str):
    """Log state transitions."""
    transition_log = logger.bind(
        conversation_id=conversation_id,
        component="STATE_TRANSITION"
    )

    transition_log.info(f"State transition: {from_state} -> {to_state} ({method})")


def log_handler_execution(
        timing: str,
        handler_name: str,
        execution_time: float,
        success: bool,
        context_updates: Dict[str, Any] = None
):
    """Log handler execution details."""
    handler_log = logger.bind(
        component="HANDLER_EXECUTION",
        performance=execution_time > 0.1
    )

    status = "SUCCESS" if success else "FAILED"
    updates_info = f", updates={list(context_updates.keys())}" if context_updates else ""

    handler_log.info(
        f"Handler {handler_name} at {timing}: {status} "
        f"(time={execution_time:.3f}s{updates_info})"
    )


def log_prompt_generation(prompt_type: str, length: int, tokens_estimated: int):
    """Log prompt generation statistics."""
    prompt_log = logger.bind(
        component="PROMPT_BUILDER",
        performance=True
    )

    prompt_log.info(
        f"Generated {prompt_type} prompt: "
        f"length={length} chars, estimated_tokens={tokens_estimated}"
    )


def log_context_update(conversation_id: str, updated_keys: List[str], context_size: int):
    """Log context updates."""
    context_log = logger.bind(
        conversation_id=conversation_id,
        component="CONTEXT_MANAGER"
    )

    context_log.debug(
        f"Context updated: keys={updated_keys}, total_size={context_size}"
    )


# --------------------------------------------------------------
# Performance and Debug Logging
# --------------------------------------------------------------

class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(self, operation_name: str, component: str = "PERFORMANCE"):
        self.operation_name = operation_name
        self.component = component
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        perf_log = logger.bind(
            component=self.component,
            performance=True
        )

        if exc_type is None:
            perf_log.info(f"{self.operation_name} completed in {duration:.3f}s")
        else:
            perf_log.error(f"{self.operation_name} failed after {duration:.3f}s: {exc_val}")


def log_debug_info(component: str, debug_data: Dict[str, Any]):
    """Log debug information."""
    debug_log = logger.bind(component=component.upper())

    try:
        debug_json = json.dumps(debug_data, indent=2, default=str)
        debug_log.debug(f"Debug info:\n{debug_json}")
    except Exception as e:
        debug_log.debug(f"Debug info (non-serializable): {debug_data}")


def log_conversation_flow(conversation_id: str, flow_data: Dict[str, Any]):
    """Log conversation flow information."""
    flow_log = logger.bind(
        conversation_id=conversation_id,
        component="CONVERSATION_FLOW"
    )

    flow_log.info(f"Conversation flow: {json.dumps(flow_data, default=str)}")


# --------------------------------------------------------------
# Configuration and Utilities
# --------------------------------------------------------------

def configure_logging(
        level: str = "INFO",
        enable_performance_logging: bool = True,
        enable_debug_files: bool = False,
        log_directory: str = "logs"
):
    """Configure logging system with custom settings."""
    global log_dir

    # Update log directory
    log_dir = Path(log_directory)
    log_dir.mkdir(exist_ok=True)

    # Remove existing handlers and reconfigure
    logger.remove()

    # Add console handler with specified level
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{extra[component]}</cyan> | {message}",
        filter=prepare_log_record
    )

    # Add main log file
    logger.add(
        log_dir / "llm-fsm.log",
        level="DEBUG" if enable_debug_files else "INFO",
        rotation="10 MB",
        retention="1 month",
        filter=prepare_log_record
    )

    if enable_performance_logging:
        logger.add(
            log_dir / "performance.log",
            level="INFO",
            filter=lambda record: record["extra"].get("performance", False),
            rotation="5 MB",
            retention="1 week"
        )


def get_logger_for_component(component: str) -> any:
    """Get logger instance bound to specific component."""
    return logger.bind(component=component.upper())


def enable_conversation_tracing(conversation_id: str):
    """Enable detailed tracing for a specific conversation."""
    trace_log = logger.bind(
        conversation_id=conversation_id,
        component="TRACE"
    )

    trace_log.info(f"Enabled detailed tracing for conversation {conversation_id}")

    return trace_log