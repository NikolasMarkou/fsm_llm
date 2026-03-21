from __future__ import annotations

"""
Enhanced FSM-LLM: Improved 2-Pass Architecture for Large Language Model Finite State Machines.

This package provides a sophisticated framework for building stateful conversational AI
systems using an improved 2-pass architecture that generates responses after transition
evaluation for optimal contextual accuracy.
"""

import sys
import warnings

from .__version__ import __version__

# --------------------------------------------------------------
# Main API Components
# --------------------------------------------------------------
from .api import API, ContextMergeStrategy

# --------------------------------------------------------------
# Core Definitions and Models
# --------------------------------------------------------------
from .definitions import (
    # Context and conversation management
    Conversation,
    # Improved 2-pass architecture models
    DataExtractionRequest,
    DataExtractionResponse,
    FSMContext,
    FSMDefinition,
    # Exception classes
    FSMError,
    FSMInstance,
    InvalidTransitionError,
    # Enums and types
    LLMRequestType,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    # Core FSM models
    State,
    StateNotFoundError,
    Transition,
    TransitionCondition,
    TransitionDecisionRequest,
    TransitionDecisionResponse,
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
# Enhanced Prompt Building Components
# --------------------------------------------------------------
from .prompts import (
    DataExtractionPromptBuilder,
    DataExtractionPromptConfig,
    ResponseGenerationPromptBuilder,
    ResponsePromptConfig,
    TransitionPromptBuilder,
    TransitionPromptConfig,
)

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
    "Conversation",

    # Improved 2-pass architecture components
    "DataExtractionRequest",
    "DataExtractionResponse",
    "ResponseGenerationRequest",
    "ResponseGenerationResponse",
    "TransitionDecisionRequest",
    "TransitionDecisionResponse",
    "TransitionOption",
    "TransitionEvaluation",
    "TransitionEvaluationResult",
    "LLMRequestType",

    # LLM interfaces
    "LLMInterface",
    "LiteLLMInterface",

    # Enhanced prompt builders
    "DataExtractionPromptBuilder",
    "ResponseGenerationPromptBuilder",
    "TransitionPromptBuilder",
    "DataExtractionPromptConfig",
    "ResponsePromptConfig",
    "TransitionPromptConfig",

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
    "HandlerSystemError",
    "HandlerExecutionError",

    # Extension checks
    "has_workflows",
    "get_workflows",
    "has_reasoning",
    "get_reasoning",
    "has_classification",
    "get_classification",
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


def has_classification():
    """Check if classification extension is available."""
    import importlib.util
    return importlib.util.find_spec("fsm_llm_classification") is not None


def get_classification():
    """Get classification module if available, otherwise raise ImportError."""
    try:
        import fsm_llm_classification
        return fsm_llm_classification
    except ImportError as e:
        raise ImportError(
            "Classification functionality requires the fsm_llm_classification package. "
            "Install with: pip install fsm-llm[classification]"
        ) from e


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
            "classification": has_classification(),
            "agents": has_agents()
        }
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

    _library_handler_ids.append(logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
        filter=prepare_log_record
    ))


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
