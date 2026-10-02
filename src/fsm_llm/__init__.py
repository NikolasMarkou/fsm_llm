"""
FSM-LLM: a 2-pass architecture for Large Language Model Finite State Machines.

This package provides a framework for building stateful conversational AI
systems using a 2-pass architecture that generates responses after transition
evaluation, so every reply comes from the state the conversation is in.
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
from .builders import APIBuilder, FSMManagerBuilder

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
    # Message-free step result
    AdvanceResult,
    # Exception classes
    BuildError,
    # Classification models
    ClassificationError,
    ClassificationExtractionConfig,
    ClassificationResponseError,
    ClassificationResult,
    ClassificationSchema,
    # Completion models (LLMInterface.complete)
    CompletionRequest,
    CompletionResponse,
    CompletionStateConfig,
    # Context and conversation management
    ContextScope,
    Conversation,
    ConversationBusyError,
    # 2-pass architecture models
    DataExtractionResponse,
    # Field extraction models
    FieldExtractionConfig,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMContext,
    FSMDefinition,
    FSMDefinitionNotFoundError,
    FSMError,
    FSMInstance,
    HierarchicalResult,
    HierarchicalSchema,
    IntentDefinition,
    IntentScore,
    InvalidTransitionError,
    # Usage counter snapshots (LiteLLMInterface.usage)
    LLMCallCounts,
    LLMResponseError,
    LLMUsage,
    ModelToolCall,
    MultiClassificationResult,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    RunBudgetExceededError,
    # Core FSM models
    State,
    StateNotFoundError,
    Transition,
    TransitionCondition,
    TransitionEvaluation,
    TransitionEvaluationError,
    TransitionEvaluationResult,
    TransitionOption,
    typed_field_extraction,
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
    clear_keys_on_entry,
    create_handler,
)

# --------------------------------------------------------------
# LLM Interface Components
# --------------------------------------------------------------
from .llm import LiteLLMEmbedder, LiteLLMInterface, LLMInterface, tool_exchange
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
from .visualizer import (
    FSMGraph,
    FSMGraphEdge,
    FSMGraphNode,
    build_fsm_graph,
    to_dot,
    to_mermaid,
    visualize_fsm_ascii,
    visualize_fsm_from_file,
)

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
    "APIBuilder",
    "FSMManagerBuilder",
    # Core definitions
    "FSMDefinition",
    "FSMInstance",
    "FSMContext",
    "State",
    "Transition",
    "TransitionCondition",
    "ContextScope",
    "CompletionStateConfig",
    "Conversation",
    # 2-pass architecture components
    "DataExtractionResponse",
    "ResponseGenerationRequest",
    "ResponseGenerationResponse",
    "TransitionOption",
    "TransitionEvaluation",
    "TransitionEvaluationResult",
    "AdvanceResult",
    # Field extraction
    "FieldExtractionConfig",
    "FieldExtractionRequest",
    "FieldExtractionResponse",
    "typed_field_extraction",
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
    "LiteLLMEmbedder",
    "CompletionRequest",
    "CompletionResponse",
    "ModelToolCall",
    "tool_exchange",
    "LLMUsage",
    "LLMCallCounts",
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
    "clear_keys_on_entry",
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
    "FSMGraph",
    "FSMGraphNode",
    "FSMGraphEdge",
    "build_fsm_graph",
    "to_mermaid",
    "to_dot",
    # Exceptions
    "FSMError",
    "BuildError",
    "FSMDefinitionNotFoundError",
    "ConversationBusyError",
    "RunBudgetExceededError",
    "StateNotFoundError",
    "InvalidTransitionError",
    "LLMResponseError",
    "TransitionEvaluationError",
    "ClassificationError",
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

# DECISION plan-2026-09-29T044048-3a032517/D-003
# The subpackages fsm_llm.{agents,reasoning,workflows,monitor,harness} are
# NOT imported here and NOT listed in __all__. They import core as
# `from fsm_llm import API`, so an eager import from this module would hit a
# partially initialised fsm_llm, and monitor would drag in fastapi on
# core-only installs. Do NOT add eager imports or a module __getattr__
# loader; `from fsm_llm import agents` already works via the submodule
# import protocol. The has_*/get_* helpers below probe the dotted names.


@lru_cache(maxsize=1)
def has_workflows():
    """Check if workflows extension is available."""
    import importlib.util

    return importlib.util.find_spec("fsm_llm.workflows") is not None


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
        "fsm_llm.workflows",
        "fsm_llm.workflows is missing from this install. "
        "Reinstall from a clone of https://github.com/NikolasMarkou/fsm_llm: pip install -e .",
    )


@lru_cache(maxsize=1)
def has_reasoning():
    """Check if reasoning extension is available."""
    import importlib.util

    return importlib.util.find_spec("fsm_llm.reasoning") is not None


def get_reasoning():
    """Get reasoning module if available, otherwise raise ImportError."""
    return _import_extension(
        "fsm_llm.reasoning",
        "fsm_llm.reasoning is missing from this install. "
        "Reinstall from a clone of https://github.com/NikolasMarkou/fsm_llm: pip install -e .",
    )


@lru_cache(maxsize=1)
def has_agents():
    """Check if agents extension is available."""
    import importlib.util

    return importlib.util.find_spec("fsm_llm.agents") is not None


def get_agents():
    """Get agents module if available, otherwise raise ImportError."""
    return _import_extension(
        "fsm_llm.agents",
        "fsm_llm.agents is missing from this install. "
        "Reinstall from a clone of https://github.com/NikolasMarkou/fsm_llm: pip install -e .",
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
        enable_library_logging,
        logger,
        prepare_log_record,
        register_stream_handler,
        reset_handlers,
    )

    # Re-enable the library loggers
    enable_library_logging()

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
    # DECISION plan-2026-09-29T044048-3a032517/D-007
    # C10: the module pattern is a prefix match, so a bare "fsm_llm" would
    # also silence lookalike siblings such as fsm_llm_contrib. Match the
    # package and its submodules only; the submodules include the five
    # subpackages (fsm_llm.agents, fsm_llm.workflows, ...), which are part of
    # the framework. Do NOT drop the "(\.|$)" boundary and do NOT add a
    # per-subpackage exclusion list.
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
