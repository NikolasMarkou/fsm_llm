"""
Enhanced FSM-LLM: Improved 2-Pass Architecture for Large Language Model Finite State Machines.

This package provides a sophisticated framework for building stateful conversational AI
systems using an improved 2-pass architecture that generates responses after transition
evaluation for optimal contextual accuracy.
"""

from __future__ import annotations

import sys
import warnings
from functools import lru_cache

from .__version__ import __version__

# --------------------------------------------------------------
# Main API Components
# --------------------------------------------------------------
from .api import API, ContextMergeStrategy

# --------------------------------------------------------------
# Core Definitions and Models
# --------------------------------------------------------------
from .classification import (
    Classifier,
    HandlerFn,
    HierarchicalClassifier,
    IntentRouter,
)

# --------------------------------------------------------------
# Context Utilities
# --------------------------------------------------------------
from .context import ContextCompactor

# --------------------------------------------------------------
# Core Definitions and Models
# --------------------------------------------------------------
from .definitions import (
    # Classification models
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationResponseError,
    ClassificationResult,
    ClassificationSchema,
    # Context and conversation management
    ContextScope,
    Conversation,
    ConversationBusyError,
    # Improved 2-pass architecture models
    DataExtractionResponse,
    # Field extraction models
    FieldExtractionConfig,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMDefinitionNotFoundError,
    # Exception classes
    FSMError,
    FSMInstance,
    HierarchicalResult,
    HierarchicalSchema,
    IntentDefinition,
    IntentScore,
    InvalidTransitionError,
    LLMResponseError,
    MultiClassificationResult,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    SchemaValidationError,
    # Core FSM models
    State,
    StateNotFoundError,
    Transition,
    TransitionCondition,
    TransitionEvaluation,
    TransitionEvaluationError,
    TransitionEvaluationResult,
    TransitionOption,
)

# --------------------------------------------------------------
# Expression Evaluation
# --------------------------------------------------------------
from .expressions import evaluate_logic
from .fsm import FSMManager

# --------------------------------------------------------------
# Handler System Components
# --------------------------------------------------------------
from .handlers import (
    BaseHandler,
    FSMHandler,
    HandlerBuilder,
    HandlerExecutionError,
    HandlerSystem,
    HandlerSystemError,
    HandlerTiming,
    create_handler,
)

# --------------------------------------------------------------
# LLM Interface Components
# --------------------------------------------------------------
from .llm import LiteLLMInterface, LLMInterface
from .logging import setup_logging

# --------------------------------------------------------------
# Working Memory
# --------------------------------------------------------------
from .memory import (
    BUFFER_CORE,
    BUFFER_ENVIRONMENT,
    BUFFER_METADATA,
    BUFFER_REASONING,
    BUFFER_SCRATCH,
    DEFAULT_BUFFERS,
    DEFAULT_HIDDEN_BUFFERS,
    WorkingMemory,
)

# --------------------------------------------------------------
# Enhanced Prompt Building Components
# --------------------------------------------------------------
from .prompts import (
    ClassificationPromptConfig,
    DataExtractionPromptBuilder,
    DataExtractionPromptConfig,
    FieldExtractionPromptBuilder,
    FieldExtractionPromptConfig,
    ResponseGenerationPromptBuilder,
    ResponsePromptConfig,
    build_classification_json_schema,
    build_classification_system_prompt,
)

# --------------------------------------------------------------
# Session Persistence
# --------------------------------------------------------------
from .session import FileSessionStore, SessionState, SessionStore

# --------------------------------------------------------------
# Transition Evaluation Components
# --------------------------------------------------------------
from .transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig

# --------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------
from .utilities import (
    extract_json_from_text,
    get_fsm_summary,
    load_fsm_definition,
    load_fsm_from_file,
)

# --------------------------------------------------------------
# Validation Components
# --------------------------------------------------------------
from .validator import FSMValidationResult, FSMValidator, validate_fsm_from_file

# --------------------------------------------------------------
# Visualization Components
# --------------------------------------------------------------
from .visualizer import visualize_fsm_ascii, visualize_fsm_from_file

# --------------------------------------------------------------
# Public API Definition
# --------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Core API
    "API",
    "ContextMergeStrategy",
    "FSMManager",
    # Core definitions
    "FSMDefinition",
    "FSMInstance",
    "FSMContext",
    "State",
    "Transition",
    "TransitionCondition",
    "ContextScope",
    "Conversation",
    # Improved 2-pass architecture components
    "DataExtractionResponse",
    "ResponseGenerationRequest",
    "ResponseGenerationResponse",
    "TransitionOption",
    "TransitionEvaluation",
    "TransitionEvaluationResult",
    # Field extraction
    "FieldExtractionConfig",
    "FieldExtractionRequest",
    "FieldExtractionResponse",
    # Classification (first-class)
    "ClassificationExtractionConfig",
    "Classifier",
    "HierarchicalClassifier",
    "IntentRouter",
    "HandlerFn",
    "IntentDefinition",
    "ClassificationSchema",
    "ClassificationResult",
    "IntentScore",
    "MultiClassificationResult",
    "HierarchicalSchema",
    "HierarchicalResult",
    "ClassificationPromptConfig",
    "build_classification_json_schema",
    "build_classification_system_prompt",
    # LLM interfaces
    "LLMInterface",
    "LiteLLMInterface",
    # Enhanced prompt builders
    "DataExtractionPromptBuilder",
    "ResponseGenerationPromptBuilder",
    "FieldExtractionPromptBuilder",
    "DataExtractionPromptConfig",
    "ResponsePromptConfig",
    "FieldExtractionPromptConfig",
    # Transition evaluation
    "TransitionEvaluator",
    "TransitionEvaluatorConfig",
    # Handler system
    "HandlerSystem",
    "FSMHandler",
    "BaseHandler",
    "HandlerBuilder",
    "HandlerTiming",
    "create_handler",
    # Context utilities
    "ContextCompactor",
    # Working memory
    "BUFFER_CORE",
    "BUFFER_ENVIRONMENT",
    "BUFFER_METADATA",
    "BUFFER_REASONING",
    "BUFFER_SCRATCH",
    "DEFAULT_BUFFERS",
    "DEFAULT_HIDDEN_BUFFERS",
    "WorkingMemory",
    # Session persistence
    "FileSessionStore",
    "SessionState",
    "SessionStore",
    # Utilities
    "load_fsm_definition",
    "load_fsm_from_file",
    "extract_json_from_text",
    "get_fsm_summary",
    "evaluate_logic",
    # Validation
    "FSMValidator",
    "validate_fsm_from_file",
    "FSMValidationResult",
    # Visualization
    "visualize_fsm_ascii",
    "visualize_fsm_from_file",
    # Exceptions
    "FSMError",
    "FSMDefinitionNotFoundError",
    "ConversationBusyError",
    "StateNotFoundError",
    "InvalidTransitionError",
    "LLMResponseError",
    "TransitionEvaluationError",
    "ClassificationError",
    "SchemaValidationError",
    "ClassificationResponseError",
    "HandlerSystemError",
    "HandlerExecutionError",
    # Extension checks
    "has_workflows",
    "get_workflows",
    "has_reasoning",
    "get_reasoning",
    "has_agents",
    "get_agents",
    # Framework info
    "get_version_info",
    # Quick start
    "quick_start",
    # Logging
    "setup_logging",
    # Debug helpers
    "enable_debug_logging",
    "disable_warnings",
]

# --------------------------------------------------------------
# Optional Extensions Check
# --------------------------------------------------------------


@lru_cache(maxsize=1)
def has_workflows():
    """Check if workflows extension is available."""
    import importlib.util

    return importlib.util.find_spec("fsm_llm_workflows") is not None


def _import_extension(package: str, hint: str):
    """Import an optional extension package, or raise ``ImportError(hint)``.

    Contract: returns the imported module. Only the package's OWN absence
    (``ModuleNotFoundError`` naming ``package`` or a parent) becomes the
    install hint; any other ``ImportError`` raised while the package imports
    (a missing third-party dependency, a bug inside the package) propagates
    unchanged (C10: it used to be rewritten as "install the extra").
    """
    import importlib

    try:
        return importlib.import_module(package)
    except ModuleNotFoundError as e:
        if e.name is None or not (
            e.name == package or package.startswith(f"{e.name}.")
        ):
            raise
        raise ImportError(hint) from e


def get_workflows():
    """Get workflows module if available, otherwise raise ImportError."""
    return _import_extension(
        "fsm_llm_workflows",
        "Workflows functionality requires the workflows extra. "
        "Install with: pip install fsm-llm[workflows]",
    )


@lru_cache(maxsize=1)
def has_reasoning():
    """Check if reasoning extension is available."""
    import importlib.util

    return importlib.util.find_spec("fsm_llm_reasoning") is not None


def get_reasoning():
    """Get reasoning module if available, otherwise raise ImportError."""
    return _import_extension(
        "fsm_llm_reasoning",
        "Reasoning functionality requires the reasoning extra. "
        "Install with: pip install fsm-llm[reasoning]",
    )


@lru_cache(maxsize=1)
def has_agents():
    """Check if agents extension is available."""
    import importlib.util

    return importlib.util.find_spec("fsm_llm_agents") is not None


def get_agents():
    """Get agents module if available, otherwise raise ImportError."""
    return _import_extension(
        "fsm_llm_agents",
        "Agents functionality requires the fsm_llm_agents package. "
        "Install with: pip install fsm-llm[agents]",
    )


# --------------------------------------------------------------
# Framework Information
# --------------------------------------------------------------


def get_version_info():
    """Get detailed version information."""
    return {
        "package_version": __version__,
        "architecture": "2-pass",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "features": {
            "data_extraction_phase": True,
            "response_generation_phase": True,
            "deterministic_transitions": True,
            "llm_assisted_transitions": True,
            "context_security": True,
            "handler_system": True,
            "fsm_stacking": True,
            "workflows": has_workflows(),
            "reasoning": has_reasoning(),
            "classification": True,
            "agents": has_agents(),
        },
    }


# --------------------------------------------------------------
# Quick Start Helper
# --------------------------------------------------------------


def quick_start(fsm_file: str, model: str | None = None) -> API:
    """
    Quick start helper for new users.

    Args:
        fsm_file: Path to FSM definition file
        model: LLM model to use

    Returns:
        Configured API instance ready to use
    """
    return API.from_file(fsm_file, model=model)


# --------------------------------------------------------------
# Development and Debug Helpers
# --------------------------------------------------------------


def enable_debug_logging():
    """Enable debug logging for development.

    Note: the dedup against a later ``setup_logging()`` call (D-013) only
    covers the EXACT ``(stderr, human, context=False)`` triple this handler
    registers itself under. A ``setup_logging()`` call requesting a
    different format (e.g. ``FSM_LLM_LOG_FORMAT=json``) or ``context=True``
    on the same stderr sink still registers as a distinct handler and still
    duplicates log lines -- by design, since those are legitimately
    different handler shapes, not a dedup gap.
    """
    from .constants import LOG_FORMAT_HUMAN, LOG_SINK_STDERR
    from .logging import (
        logger,
        prepare_log_record,
        register_stream_handler,
        reset_handlers,
    )

    # Re-enable the library loggers
    logger.enable("fsm_llm")

    # Only remove library-registered handlers (not user's handlers), and let
    # setup_file_logging be called again.
    reset_handlers()

    handler_id = logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
        filter=prepare_log_record,
    )

    # DECISION plan-2026-09-20T114608-a8e47b88/D-013
    # Register this handler under setup_logging()'s own (sink, format,
    # context) key. Without this, a later setup_logging(sink="stderr",
    # format="human") call cannot see this handler already exists and adds a
    # SECOND stderr handler, duplicating every subsequent log line.
    # context=False matches this handler's own (non-contextual) format string
    # above -- do NOT pass a different triple. register_stream_handler builds
    # the key with setup_logging()'s own _stream_key, so the shapes cannot drift.
    register_stream_handler(handler_id, LOG_SINK_STDERR, LOG_FORMAT_HUMAN, False)


def disable_warnings():
    """Disable framework warnings."""
    # C10: the module pattern is a prefix match, so a bare "fsm_llm" also
    # silenced fsm_llm_agents, fsm_llm_workflows, ... Match the package and
    # its submodules only.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"fsm_llm(\.|$)")


# --------------------------------------------------------------
# Module Metadata
# --------------------------------------------------------------

__title__ = "fsm-llm"
__description__ = "Finite State Machines infused with Large Language Models"
__url__ = "https://github.com/NikolasMarkou/fsm_llm"
__author__ = "Nikolas Markou"
__email__ = "nikolasmarkou@gmail.com"
__license__ = "Apache-2.0"
__copyright__ = "Copyright 2025"
