"""
Enhanced FSM-LLM: Improved 2-Pass Architecture for Large Language Model Finite State Machines.

This package provides a sophisticated framework for building stateful conversational AI
systems using an improved 2-pass architecture that generates responses after transition
evaluation for optimal contextual accuracy.
"""

import sys
import warnings

from .__version__ import __version__
from .constants import MIGRATION_WARNINGS_ENABLED

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

from .api import API, ContextMergeStrategy
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
    "HandlerExecutionError"
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
    except ImportError:
        raise ImportError(
            "Workflows functionality requires the workflows extra. "
            "Install with: pip install fsm-llm[workflows]"
        )


def has_reasoning():
    """Check if reasoning extension is available."""
    import importlib.util
    return importlib.util.find_spec("fsm_llm_reasoning") is not None


def get_reasoning():
    """Get reasoning module if available, otherwise raise ImportError."""
    try:
        import fsm_llm_reasoning
        return fsm_llm_reasoning
    except ImportError:
        raise ImportError(
            "Reasoning functionality requires the reasoning extra. "
            "Install with: pip install fsm-llm[reasoning]"
        )


def has_classification():
    """Check if classification extension is available."""
    import importlib.util
    return importlib.util.find_spec("fsm_llm_classification") is not None


def get_classification():
    """Get classification module if available, otherwise raise ImportError."""
    try:
        import fsm_llm_classification
        return fsm_llm_classification
    except ImportError:
        raise ImportError(
            "Classification functionality requires the fsm_llm_classification package. "
            "Install with: pip install fsm-llm[classification]"
        )


# Add extension check functions to public API
__all__.extend([
    "has_workflows",
    "get_workflows",
    "has_reasoning",
    "get_reasoning",
    "has_classification",
    "get_classification"
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
            "reasoning": has_reasoning(),
            "classification": has_classification()
        }
    }


def get_feature_flags():
    """Get current feature flag status."""
    from .constants import (
        ENABLE_TRANSITION_CACHING,
        ENABLE_PROMPT_OPTIMIZATION,
        ENABLE_CONTEXT_COMPRESSION,
        ENABLE_PARALLEL_EVALUATION,
        ENABLE_SMART_FALLBACKS,
        USE_ENHANCED_JSONLOGIC,
        USE_SMART_CONTEXT_FILTERING,
        USE_ADAPTIVE_THRESHOLDS
    )

    return {
        "transition_caching": ENABLE_TRANSITION_CACHING,
        "prompt_optimization": ENABLE_PROMPT_OPTIMIZATION,
        "context_compression": ENABLE_CONTEXT_COMPRESSION,
        "parallel_evaluation": ENABLE_PARALLEL_EVALUATION,
        "smart_fallbacks": ENABLE_SMART_FALLBACKS,
        "enhanced_jsonlogic": USE_ENHANCED_JSONLOGIC,
        "smart_context_filtering": USE_SMART_CONTEXT_FILTERING,
        "adaptive_thresholds": USE_ADAPTIVE_THRESHOLDS
    }


# Add info functions to public API
__all__.extend([
    "get_version_info",
    "get_feature_flags"
])

# --------------------------------------------------------------
# Import Validation and Warnings
# --------------------------------------------------------------

# Check Python version
if sys.version_info < (3, 10):
    warnings.warn(
        "FSM-LLM requires Python 3.10 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}",
        RuntimeWarning
    )

# Migration warnings for V4.0 users
if MIGRATION_WARNINGS_ENABLED:
    def _check_legacy_usage():
        """Check for legacy usage patterns and warn users."""
        # This could be expanded to detect legacy patterns
        pass

    # Could add deprecation warnings here for removed features


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
    """Enable debug logging for development."""
    from .logging import logger, _library_handler_ids

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
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}"
    ))


def disable_warnings():
    """Disable framework warnings."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=r"fsm_llm")


__all__.extend([
    "enable_debug_logging",
    "disable_warnings"
])

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
