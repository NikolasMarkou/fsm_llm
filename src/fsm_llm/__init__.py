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
# Context Utilities
# --------------------------------------------------------------
from .context import ContextCompactor

# --------------------------------------------------------------
# Main API Components
# --------------------------------------------------------------
# `API` is served via module-level ``__getattr__`` (defined at the bottom of
# this file) so that ``from fsm_llm import API`` emits a DeprecationWarning
# at the I5 epoch (since 0.6.0; removal 0.7.0). The replacement is the
# unified ``Program`` facade. ``ContextMergeStrategy`` is not deprecated.
from .dialog.api import API as _API_INTERNAL
from .dialog.api import ContextMergeStrategy

# --------------------------------------------------------------
# Core Definitions and Models
# --------------------------------------------------------------
from .dialog.classification import (
    Classifier,
    HandlerFn,
    HierarchicalClassifier,
    IntentRouter,
)

# --------------------------------------------------------------
# FSM compiler — top-level convenience (R11). Lives in dialog/.
# --------------------------------------------------------------
from .dialog.compile_fsm import compile_fsm, compile_fsm_cached

# --------------------------------------------------------------
# Core Definitions and Models
# --------------------------------------------------------------
from .dialog.definitions import (
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
from .dialog.fsm import FSMManager
from .dialog.prompts import (
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
# Expression Evaluation
# --------------------------------------------------------------
from .expressions import evaluate_logic

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
from .logging import setup_logging

# --------------------------------------------------------------
# Working Memory
# --------------------------------------------------------------
from .memory import BUFFER_METADATA, WorkingMemory
from .profiles import (
    HarnessProfile,
    ProviderProfile,
    get_harness_profile,
    get_provider_profile,
    register_harness_profile,
    register_provider_profile,
)

# --------------------------------------------------------------
# Program facade (R1 + R8) — unified entry point
# --------------------------------------------------------------
from .program import ExplainOutput, Program, ProgramModeError, Result

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
# LLM Interface Components
# --------------------------------------------------------------
from .runtime._litellm import (  # noqa: F401  D-009: LiteLLMInterface intentionally NOT in __all__ but kept importable for back-compat
    LiteLLMInterface,
    LLMInterface,
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
    aggregate_term,
    multi_hop_dynamic_term,
    multi_hop_term,
    niah_padded_term,
    niah_term,
    pairwise_term,
)
from .stdlib.reasoning.lam_factories import (
    abductive_term,
    analogical_term,
    analytical_term,
    calculator_term,
    classifier_term,
    creative_term,
    critical_term,
    deductive_term,
    hybrid_term,
    inductive_term,
    solve_term,
)
from .stdlib.workflows.lam_factories import (
    branch_term,
    linear_term,
    parallel_term,
    retry_term,
    switch_term,
)

# `Handler` is the (R11) top-level alias for `FSMHandler` — the legacy
# name remains exported for back-compat. New code should prefer
# `from fsm_llm import Handler`.
Handler = FSMHandler

# --------------------------------------------------------------
# Session Persistence
# --------------------------------------------------------------
from .dialog.session import FileSessionStore, SessionState, SessionStore

# --------------------------------------------------------------
# Transition Evaluation Components
# --------------------------------------------------------------
from .dialog.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig

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
    # L2 COMPOSE — handler/composition surface (M2 layer-explicit).
    # 0.6.0: full handler surface (HandlerSystem / FSMHandler /
    # BaseHandler / create_handler) is in L2, not Legacy — users
    # wiring handlers no longer need to reach into Legacy.
    # ----------------------------------------------------------------
    "compose",
    "Handler",
    "HandlerTiming",
    "HandlerBuilder",
    "HandlerSystem",
    "FSMHandler",
    "BaseHandler",
    "create_handler",
    # Profiles (L2 COMPOSE) — construction-time data bundles applied
    # apply-once at Program.from_*. See `src/fsm_llm/profiles.py` and
    # `docs/api_reference.md` Profiles section.
    "HarnessProfile",
    "ProviderProfile",
    "register_harness_profile",
    "register_provider_profile",
    "get_harness_profile",
    "get_provider_profile",
    # ----------------------------------------------------------------
    # L3 AUTHOR — Stdlib factory terms (R11) + FSM compiler.
    # Top-level convenience for term-mode authoring. All stdlib
    # factories follow the ``*_term`` convention.
    # ----------------------------------------------------------------
    # Agents (4)
    "react_term",
    "rewoo_term",
    "reflexion_term",
    "memory_term",
    # Reasoning (11)
    "analytical_term",
    "deductive_term",
    "inductive_term",
    "abductive_term",
    "analogical_term",
    "creative_term",
    "critical_term",
    "hybrid_term",
    "calculator_term",
    "classifier_term",
    "solve_term",
    # Workflows (5)
    "linear_term",
    "branch_term",
    "switch_term",
    "parallel_term",
    "retry_term",
    # Long-context (6)
    "niah_term",
    "aggregate_term",
    "pairwise_term",
    "multi_hop_term",
    "multi_hop_dynamic_term",
    "niah_padded_term",
    # FSM compiler
    "compile_fsm",
    "compile_fsm_cached",
    # ----------------------------------------------------------------
    # Legacy — FSM dialog front-end + utilities (silent shims; predates
    # M2 layer partition. Excluded from _LAYER_L1..L4 by design.)
    #
    # Sub-partition (0.6.0): a single name in this block is on the
    # deprecation calendar — `API`. Accessing `fsm_llm.API` warns
    # (since=0.6.0, removal=0.7.0); replacement is `Program.from_fsm`.
    # All other Legacy names are supported and not scheduled for removal.
    # The `_LAYER_LEGACY_DEPRECATED` frozenset below codifies this so
    # the deprecation calendar test can audit it.
    # ----------------------------------------------------------------
    # Deprecated, removal=0.7.0 (see __getattr__ at module bottom).
    "API",
    # Supported Legacy — FSM dialog surface, classification, prompts,
    # transition evaluation, sessions, validators, exceptions.
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
# Layer partition (M2 — merge spec §4 CAND-E + §6 G3)
#
# `_LAYER_L1.._LAYER_L4` partition the layer-explicit subset of
# `__all__`. The Legacy block is the complement
# (`set(__all__) - (L1 | L2 | L3 | L4)`) — no `_LAYER_LEGACY` is
# stored, to minimise drift surface. The layering invariant
# (disjoint + cover) is asserted by
# `tests/test_fsm_llm/test_layering.py`.
#
# `_LAYER_LEGACY_DEPRECATED` (0.6.0) names the strict subset of
# the Legacy block that is on the deprecation calendar — accessing
# any of these from `fsm_llm` warns and is removed at the indicated
# version. The deprecation calendar test asserts membership.
#
# These frozensets are PRIVATE (underscore-prefixed). They are
# NOT in `__all__`. Their sole consumer is the layering audit
# test; their meaning is documented in `docs/lambda_fsm_merge.md` §3 I4.
# --------------------------------------------------------------

_LAYER_L4: frozenset[str] = frozenset(
    {
        "Program",
        "Result",
        "ExplainOutput",
        "ProgramModeError",
    }
)

# Legacy entries scheduled for removal. Subset of the Legacy block
# (set(__all__) - (L1|L2|L3|L4)). Each name accessed from `fsm_llm`
# emits DeprecationWarning via the module-level `__getattr__`.
_LAYER_LEGACY_DEPRECATED: frozenset[str] = frozenset(
    {
        "API",  # since=0.6.0, removal=0.7.0; replacement: Program.from_fsm
    }
)

_LAYER_L3: frozenset[str] = frozenset(
    {
        # Agents (4)
        "react_term",
        "rewoo_term",
        "reflexion_term",
        "memory_term",
        # Reasoning (11)
        "analytical_term",
        "deductive_term",
        "inductive_term",
        "abductive_term",
        "analogical_term",
        "creative_term",
        "critical_term",
        "hybrid_term",
        "calculator_term",
        "classifier_term",
        "solve_term",
        # Workflows (5)
        "linear_term",
        "branch_term",
        "switch_term",
        "parallel_term",
        "retry_term",
        # Long-context (6) — *_term canonical (0.6.0+)
        "niah_term",
        "aggregate_term",
        "pairwise_term",
        "multi_hop_term",
        "multi_hop_dynamic_term",
        "niah_padded_term",
        # FSM compiler (R11 top-level shortcut)
        "compile_fsm",
        "compile_fsm_cached",
    }
)

_LAYER_L2: frozenset[str] = frozenset(
    {
        # Composition surface
        "compose",
        # Handler API (0.6.0: full surface lifted out of Legacy)
        "Handler",
        "FSMHandler",
        "BaseHandler",
        "HandlerTiming",
        "HandlerBuilder",
        "HandlerSystem",
        "create_handler",
        # Profiles — construction-time data bundles applied via
        # apply_to_term (Term -> Term) at Program.from_*. Pure
        # AST-side; touches Leaf.template only via model_copy.
        "HarnessProfile",
        "ProviderProfile",
        "register_harness_profile",
        "register_provider_profile",
        "get_harness_profile",
        "get_provider_profile",
    }
)

_LAYER_L1: frozenset[str] = frozenset(
    {
        # AST node types
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
    }
)

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
        from fsm_llm.stdlib import workflows

        return workflows
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
        from fsm_llm.stdlib import reasoning

        return reasoning
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
        from fsm_llm.stdlib import agents

        return agents
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


def quick_start(fsm_file: str, model: str | None = None) -> _API_INTERNAL:
    """
    Quick start helper for new users.

    Args:
        fsm_file: Path to FSM definition file
        model: LLM model to use

    Returns:
        Configured API instance ready to use
    """
    return _API_INTERNAL.from_file(fsm_file, model=model)


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


# --------------------------------------------------------------
# I5 deprecation: top-level legacy ``API`` re-export
# --------------------------------------------------------------
#
# Per ``docs/lambda_fsm_merge.md`` §3 I5 (two-epoch reconciliation), the
# top-level ``from fsm_llm import API`` re-export warns at 0.6.0 and is
# removed at 0.7.0. The replacement is the unified ``Program`` facade
# (``Program.from_fsm``). Served via module-level ``__getattr__`` so that
# accessing ``fsm_llm.API`` or ``from fsm_llm import API`` triggers the
# warning exactly once per process (deduped by ``warn_deprecated``).
def __getattr__(name: str):  # PEP 562
    if name == "API":
        from ._api.deprecation import warn_deprecated

        warn_deprecated(
            "fsm_llm.API",
            since="0.6.0",
            removal="0.7.0",
            replacement="Program.from_fsm",
        )
        return _API_INTERNAL
    raise AttributeError(f"module 'fsm_llm' has no attribute {name!r}")
