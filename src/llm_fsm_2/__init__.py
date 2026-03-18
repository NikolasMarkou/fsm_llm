"""
Enhanced LLM-FSM: Improved 2-Pass Architecture for Large Language Model Finite State Machines.

This package provides a sophisticated framework for building stateful conversational AI
systems using an improved 2-pass architecture that generates responses after transition
evaluation for optimal contextual accuracy.
"""

import sys
import warnings

from .__version__ import __version__

# --------------------------------------------------------------
# Core Definitions and Models
# --------------------------------------------------------------

from .definitions import (
    # Core FSM models
    State,
    Transition,
    TransitionCondition,
    FSMDefinition,
    FSMInstance,
    FSMContext,

    # Context and conversation management
    Conversation,

    # Improved 2-pass architecture models
    DataExtractionRequest,
    DataExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    TransitionDecisionRequest,
    TransitionDecisionResponse,
    TransitionOption,
    TransitionEvaluation,
    TransitionEvaluationResult,

    # Enums and types
    LLMRequestType,

    # Exception classes
    FSMError,
    StateNotFoundError,
    InvalidTransitionError,
    LLMResponseError,
    TransitionEvaluationError
)

# --------------------------------------------------------------
# Main API Components
# --------------------------------------------------------------

from .api import API
from .fsm import FSMManager

# --------------------------------------------------------------
# LLM Interface Components
# --------------------------------------------------------------

from .llm import LLMInterface, LiteLLMInterface

# --------------------------------------------------------------
# Enhanced Prompt Building Components
# --------------------------------------------------------------

from .prompts import (
    DataExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
    TransitionPromptBuilder,
    DataExtractionPromptConfig,
    ResponsePromptConfig,
    TransitionPromptConfig,
)

# --------------------------------------------------------------
# Transition Evaluation Components
# --------------------------------------------------------------

from .transition_evaluator import (
    TransitionEvaluator,
    TransitionEvaluatorConfig
)

# --------------------------------------------------------------
# Handler System Components
# --------------------------------------------------------------

from .handlers import (
    HandlerSystem,
    FSMHandler,
    BaseHandler,
    HandlerBuilder,
    HandlerTiming,
    create_handler,
    HandlerSystemError,
    HandlerExecutionError
)

# --------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------

from .utilities import (
    load_fsm_definition,
    load_fsm_from_file,
    extract_json_from_text,
    validate_json_structure,
    get_fsm_summary
)

# --------------------------------------------------------------
# Expression Evaluation
# --------------------------------------------------------------

from .expressions import evaluate_logic

# --------------------------------------------------------------
# Validation Components
# --------------------------------------------------------------

from .validator import (
    FSMValidator,
    validate_fsm_from_file,
    FSMValidationResult
)

# --------------------------------------------------------------
# Visualization Components
# --------------------------------------------------------------

from .visualizer import (
    visualize_fsm_ascii,
    visualize_fsm_from_file
)

# --------------------------------------------------------------
# Public API Definition
# --------------------------------------------------------------

__all__ = [
    # Version
    "__version__",

    # Core API
    "API",
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
    "HandlerExecutionError"
]

# --------------------------------------------------------------
# Optional Extensions Check
# --------------------------------------------------------------

def has_workflows():
    """Check if workflows extension is available."""
    try:
        import llm_fsm_workflows
        return True
    except ImportError:
        return False


def get_workflows():
    """Get workflows module if available, otherwise raise ImportError."""
    try:
        import llm_fsm_workflows
        return llm_fsm_workflows
    except ImportError:
        raise ImportError(
            "Workflows functionality requires the workflows extra. "
            "Install with: pip install llm-fsm[workflows]"
        )


def has_reasoning():
    """Check if reasoning extension is available."""
    try:
        import llm_fsm_reasoning
        return True
    except ImportError:
        return False


def get_reasoning():
    """Get reasoning module if available, otherwise raise ImportError."""
    try:
        import llm_fsm_reasoning
        return llm_fsm_reasoning
    except ImportError:
        raise ImportError(
            "Reasoning functionality requires the reasoning extra. "
            "Install with: pip install llm-fsm[reasoning]"
        )


# Add extension check functions to public API
__all__.extend([
    "has_workflows",
    "get_workflows",
    "has_reasoning",
    "get_reasoning"
])


# --------------------------------------------------------------
# Framework Information
# --------------------------------------------------------------

def get_version_info():
    """Get detailed version information."""
    from .constants import FRAMEWORK_VERSION, API_VERSION

    return {
        "framework_version": FRAMEWORK_VERSION,
        "api_version": API_VERSION,
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
            "reasoning": has_reasoning()
        }
    }


# Add info functions to public API
__all__.extend([
    "get_version_info",
])

# --------------------------------------------------------------
# Import Validation and Warnings
# --------------------------------------------------------------

# Check Python version
if sys.version_info < (3, 10):
    warnings.warn(
        "LLM-FSM requires Python 3.10 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}",
        RuntimeWarning
    )


# --------------------------------------------------------------
# Quick Start Helper
# --------------------------------------------------------------

def quick_start(fsm_file: str, model: str = "gpt-4o-mini") -> API:
    """
    Quick start helper for new users.

    Args:
        fsm_file: Path to FSM definition file
        model: LLM model to use

    Returns:
        Configured API instance ready to use
    """
    return API.from_file(fsm_file, model=model)


__all__.append("quick_start")


# --------------------------------------------------------------
# Development and Debug Helpers
# --------------------------------------------------------------

def enable_debug_logging():
    """Enable debug logging for development. Only removes library-added handlers."""
    from . import logging as log_module
    from .logging import logger, prepare_log_record

    # Remove only library-added handlers (not user-registered ones)
    for handler_id in log_module._library_handler_ids:
        try:
            logger.remove(handler_id)
        except ValueError:
            pass  # Already removed
    log_module._library_handler_ids.clear()

    log_module._file_handler_initialized = False  # Allow file logging to be re-setup
    handler_id = logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
        filter=prepare_log_record
    )
    log_module._library_handler_ids.append(handler_id)


def disable_warnings():
    """Disable framework warnings."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"llm_fsm(_2)?")


__all__.extend([
    "enable_debug_logging",
    "disable_warnings"
])

# --------------------------------------------------------------
# Module Metadata
# --------------------------------------------------------------

__title__ = "llm-fsm"
__description__ = "Finite State Machines infused with Large Language Models"
__url__ = "https://github.com/NikolasMarkou/fsm_llm"
__author__ = "Nikolas Markou"
__email__ = "nikolasmarkou@gmail.com"
__license__ = "GPLv3"
__copyright__ = "Copyright 2025"