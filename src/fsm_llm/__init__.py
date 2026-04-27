from __future__ import annotations

"""
Enhanced FSM-LLM: Improved 2-Pass Architecture for Large Language Model Finite State Machines.

This package provides a sophisticated framework for building stateful conversational AI
systems using an improved 2-pass architecture that generates responses after transition
evaluation for optimal contextual accuracy.
"""

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
    Conversation,
    # Improved 2-pass architecture models
    DataExtractionResponse,
    DomainSchema,
    # Field extraction models
    FieldExtractionConfig,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMContext,
    FSMDefinition,
    # Exception classes
    FSMError,
    FSMInstance,
    HierarchicalResult,
    HierarchicalSchema,
    IntentDefinition,
    IntentScore,
    InvalidTransitionError,
    # Enums and types
    LLMRequestType,
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
# FSM compiler — top-level convenience (R11). Lives in dialog/.
# --------------------------------------------------------------
from .dialog.compile_fsm import compile_fsm, compile_fsm_cached

# --------------------------------------------------------------
# Expression Evaluation
# --------------------------------------------------------------
from .expressions import evaluate_logic
from .fsm import FSMManager

# --------------------------------------------------------------
# Handler System Components
# --------------------------------------------------------------
# --------------------------------------------------------------
# Handlers — `Handler` alias for `FSMHandler` (R11 nicer naming)
# and `compose` for term-mode handler splicing.
# --------------------------------------------------------------
from .handlers import (
    BaseHandler,
    FSMHandler,
    HandlerBuilder,
    HandlerExecutionError,
    HandlerSystem,
    HandlerSystemError,
    HandlerTiming,
    compose,
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
from .memory import BUFFER_METADATA, WorkingMemory

# --------------------------------------------------------------
# Program facade (R1 + R8) — unified entry point
# --------------------------------------------------------------
from .program import ExplainOutput, Program, ProgramModeError, Result
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
# λ-substrate kernel (R11 promotion) — first-class at top-level.
# Substrate names appear before FSM-front-end names in __all__ so the
# substrate-as-primary positioning is visible at `from fsm_llm import …`.
# --------------------------------------------------------------
from .runtime import (
    BUILTIN_OPS,
    Abs,
    App,
    ASTConstructionError,
    Case,
    Combinator,
    CombinatorOp,
    CostAccumulator,
    Executor,
    Fix,
    LambdaError,
    Leaf,
    LeafCall,
    Let,
    LiteLLMOracle,
    Oracle,
    OracleError,
    Plan,
    PlanInputs,
    PlanningError,
    ReduceOp,
    Term,
    TerminationError,
    Var,
    abs_,
    app,
    case_,
    concat,
    cross,
    ffilter,
    fix,
    fmap,
    host_call,
    is_term,
    leaf,
    let_,
    peek,
    plan,
    reduce_,
    split,
    var,
)

# --------------------------------------------------------------
# Stdlib factory terms (R11) — convenience exports for the most-used
# named factories. Full surface available under fsm_llm.stdlib.*.
# --------------------------------------------------------------
from .stdlib.agents import (
    memory_term,
    react_term,
    reflexion_term,
    rewoo_term,
)
from .stdlib.long_context import (
    aggregate,
    multi_hop,
    niah,
    pairwise,
)

# `Handler` is the (R11) top-level alias for `FSMHandler` — the legacy
# name remains exported for back-compat. New code should prefer
# `from fsm_llm import Handler`.
Handler = FSMHandler

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
    validate_json_structure,
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
    # ----------------------------------------------------------------
    # Program facade (R1 + R8) — single user-visible execution verb
    # ----------------------------------------------------------------
    "Program",
    "Result",
    "ExplainOutput",
    "ProgramModeError",
    # ----------------------------------------------------------------
    # λ-substrate kernel (R11) — substrate names are first-class.
    # Importing `from fsm_llm import Term, leaf, fix, Executor, …`
    # is the recommended path for term-mode authoring. Full surface
    # remains importable from `fsm_llm.runtime` and `fsm_llm.lam` (shim).
    # ----------------------------------------------------------------
    "Term",
    "Var",
    "Abs",
    "App",
    "Let",
    "Case",
    "Combinator",
    "CombinatorOp",
    "Fix",
    "Leaf",
    "is_term",
    # DSL builders
    "var",
    "abs_",
    "app",
    "let_",
    "case_",
    "fix",
    "leaf",
    "split",
    "peek",
    "fmap",
    "ffilter",
    "reduce_",
    "concat",
    "cross",
    "host_call",
    # Combinators
    "ReduceOp",
    "BUILTIN_OPS",
    # Planner
    "PlanInputs",
    "Plan",
    "plan",
    # Oracle + cost
    "Oracle",
    "LiteLLMOracle",
    "Executor",
    "LeafCall",
    "CostAccumulator",
    # Kernel exceptions
    "LambdaError",
    "ASTConstructionError",
    "TerminationError",
    "PlanningError",
    "OracleError",
    # ----------------------------------------------------------------
    # Stdlib factory terms (R11) — top-level convenience.
    # ----------------------------------------------------------------
    "react_term",
    "rewoo_term",
    "reflexion_term",
    "memory_term",
    "niah",
    "aggregate",
    "pairwise",
    "multi_hop",
    # ----------------------------------------------------------------
    # FSM compiler (R11) — top-level shortcut to dialog.compile_fsm.
    # ----------------------------------------------------------------
    "compile_fsm",
    "compile_fsm_cached",
    # ----------------------------------------------------------------
    # Core API (FSM dialog front-end)
    # ----------------------------------------------------------------
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
    "Conversation",
    # Improved 2-pass architecture components
    "DataExtractionResponse",
    "ResponseGenerationRequest",
    "ResponseGenerationResponse",
    "TransitionOption",
    "TransitionEvaluation",
    "TransitionEvaluationResult",
    "LLMRequestType",
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
    "DomainSchema",
    "HierarchicalSchema",
    "HierarchicalResult",
    "ClassificationPromptConfig",
    "build_classification_json_schema",
    "build_classification_system_prompt",
    # LLM interfaces
    "LLMInterface",
    # DECISION D-009 (R10 step 8): LiteLLMInterface un-exported from
    # fsm_llm.__all__ — it is the private adapter behind the Oracle layer
    # post-R10. Direct construction still works via
    # `from fsm_llm.runtime._litellm import LiteLLMInterface` for the
    # 3 deferred dialog-site legacy paths and for back-compat tests, but
    # new code must compose through `LiteLLMOracle(llm)` (preferred) or
    # `from fsm_llm import Program` and let the facade pick the oracle.
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
    "Handler",
    "BaseHandler",
    "HandlerBuilder",
    "HandlerTiming",
    "create_handler",
    "compose",
    # Context utilities
    "ContextCompactor",
    # Working memory
    "BUFFER_METADATA",
    "WorkingMemory",
    # Session persistence
    "FileSessionStore",
    "SessionState",
    "SessionStore",
    # Utilities
    "load_fsm_definition",
    "load_fsm_from_file",
    "extract_json_from_text",
    "validate_json_structure",
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


def get_workflows():
    """Get workflows module if available, otherwise raise ImportError."""
    try:
        import fsm_llm_workflows

        return fsm_llm_workflows
    except ImportError as e:
        raise ImportError(
            "Workflows functionality requires the workflows extra. "
            "Install with: pip install fsm-llm[workflows]"
        ) from e


@lru_cache(maxsize=1)
def has_reasoning():
    """Check if reasoning extension is available."""
    import importlib.util

    return importlib.util.find_spec("fsm_llm_reasoning") is not None


def get_reasoning():
    """Get reasoning module if available, otherwise raise ImportError."""
    try:
        import fsm_llm_reasoning

        return fsm_llm_reasoning
    except ImportError as e:
        raise ImportError(
            "Reasoning functionality requires the reasoning extra. "
            "Install with: pip install fsm-llm[reasoning]"
        ) from e


@lru_cache(maxsize=1)
def has_agents():
    """Check if agents extension is available."""
    import importlib.util

    return importlib.util.find_spec("fsm_llm_agents") is not None


def get_agents():
    """Get agents module if available, otherwise raise ImportError."""
    try:
        import fsm_llm_agents

        return fsm_llm_agents
    except ImportError as e:
        raise ImportError(
            "Agents functionality requires the fsm_llm_agents package. "
            "Install with: pip install fsm-llm[agents]"
        ) from e


# --------------------------------------------------------------
# Framework Information
# --------------------------------------------------------------


def get_version_info():
    """Get detailed version information."""
    return {
        "package_version": __version__,
        "architecture": "improved-2-pass",
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


def disable_warnings():
    """Disable framework warnings."""
    warnings.filterwarnings("ignore", category=UserWarning, module=r"fsm_llm")


# --------------------------------------------------------------
# Module Metadata
# --------------------------------------------------------------

__title__ = "fsm-llm"
__description__ = "Finite State Machines infused with Large Language Models"
__url__ = "https://github.com/NikolasMarkou/fsm_llm"
__author__ = "Nikolas Markou"
__email__ = "nikolasmarkou@gmail.com"
__license__ = "GPLv3"
__copyright__ = "Copyright 2025"
