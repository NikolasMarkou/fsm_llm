from __future__ import annotations

"""
fsm_llm — Stateful LLM programs on a typed λ-calculus runtime.

Two surface syntaxes share one executor:

- **FSM JSON (Category A)** — dialog programs with persistent per-turn state.
  Compiled to λ-terms at load time via ``compile_fsm``.
- **λ-DSL (Category B/C)** — pipelines, agents, reasoning chains, long-context
  recursion. Authored as λ-terms directly via ``fsm_llm.dsl`` /
  ``fsm_llm.factories``.

Both surfaces flow through one verb: ``Program.invoke(...) → Result``.

Top-level imports give you the high-traffic API. Sub-namespaces hold the
substrate:

    from fsm_llm import Program, compile_fsm, Executor, FSMHandler
    from fsm_llm.ast import Term, Var, Abs, App, Let, Case, Leaf, Fix
    from fsm_llm.dsl import leaf, var, abs_, app, let, case_, fix
    from fsm_llm.combinators import split, fmap, ffilter, reduce
    from fsm_llm.factories import react_term, analytical_term, linear_term
    from fsm_llm.errors import FSMError, LambdaError, ProgramModeError
    from fsm_llm.debug import enable_debug_logging

See ``docs/lambda_fsm_merge.md`` for the merge contract and
``docs/lambda.md`` for the architectural thesis.
"""

from .__version__ import __version__

# --- Pydantic request/response models ---
# Re-import FSMError after dialog tier loaded (it's already imported above
# via _models for the Pydantic block; this is a no-op).
from ._models import (
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMError,
    LLMRequestType,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    TransitionEvaluationResult,
)

# --- Context & memory ---
from .context import ContextCompactor

# ----------------------------------------------------------------------------
# Load dialog tier first to avoid circular imports between runtime/_litellm
# and dialog/api (the LiteLLMInterface ↔ utilities ↔ dialog.definitions
# triangle resolves cleanly when dialog initialises before runtime).
# ============================================================================
# Dialog tier — FSM dialog front-end. These names stay top-level (stable,
# supported, not deprecated). The "Legacy" label is retired at 0.9.0; below
# they are organised into thematic groups.
# ============================================================================
# --- FSM dialog core ---
from .dialog.api import ContextMergeStrategy

# --- Classification & extraction ---
from .dialog.classification import (
    Classifier,
    HandlerFn,
    HierarchicalClassifier,
    IntentRouter,
)

# ----------------------------------------------------------------------------
# FSM compiler — top-level convenience. Canonical home is fsm_llm.dialog.
# ----------------------------------------------------------------------------
from .dialog.compile_fsm import compile_fsm
from .dialog.definitions import (
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    Conversation,
    DomainSchema,
    FieldExtractionConfig,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    HierarchicalResult,
    HierarchicalSchema,
    IntentDefinition,
    IntentScore,
    MultiClassificationResult,
    State,
    Transition,
    TransitionCondition,
    TransitionEvaluation,
    TransitionOption,
)
from .dialog.fsm import FSMManager

# --- Prompt builders ---
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

# --- Session persistence ---
from .dialog.session import FileSessionStore, SessionState, SessionStore

# --- Transition evaluation ---
from .dialog.transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig

# --- Validation, visualization, loaders ---
from .expressions import evaluate_logic

# ----------------------------------------------------------------------------
# Handler surface — composition + builder + protocols.
# ----------------------------------------------------------------------------
from .handlers import (
    BaseHandler,
    FSMHandler,
    HandlerBuilder,
    HandlerTiming,
    compose,
    create_handler,
)

# ----------------------------------------------------------------------------
# Logging — public setup helper.
# ----------------------------------------------------------------------------
from .logging import setup_logging
from .memory import WorkingMemory

# ----------------------------------------------------------------------------
# Profiles — construction-time data bundles applied apply-once at Program.from_*.
# ----------------------------------------------------------------------------
from .profiles import (
    HarnessProfile,
    ProfileRegistry,
    ProviderProfile,
    profile_registry,
)

# ----------------------------------------------------------------------------
# Program facade — one verb, three constructors. The user-visible entry point.
# Imported last because it depends on the L1-L3 stack being initialised.
# ----------------------------------------------------------------------------
from .program import ExplainOutput, Program, ProgramModeError, Result

# ----------------------------------------------------------------------------
# λ-substrate — top-level convenience for the most-used kernel names.
# Full surface in fsm_llm.ast / fsm_llm.dsl / fsm_llm.combinators / fsm_llm.runtime.
# ----------------------------------------------------------------------------
from .runtime import (
    CostAccumulator,
    Executor,
    LiteLLMOracle,
    Oracle,
    Plan,
    PlanInputs,
    plan,
)

# ----------------------------------------------------------------------------
# Root errors — full hierarchy at fsm_llm.errors.
# ----------------------------------------------------------------------------
from .runtime.errors import LambdaError
from .utilities import (
    extract_json_from_text,
    get_fsm_summary,
    load_fsm_definition,
    load_fsm_from_file,
    validate_json_structure,
)
from .validator import FSMValidationResult, FSMValidator, validate_fsm_from_file
from .visualizer import visualize_fsm_ascii, visualize_fsm_from_file

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

__all__ = [
    "__version__",
    # ----- Program facade -----
    "Program",
    "Result",
    "ExplainOutput",
    "ProgramModeError",
    # ----- λ-substrate (top-level convenience) -----
    "Executor",
    "Plan",
    "PlanInputs",
    "plan",
    "Oracle",
    "LiteLLMOracle",
    "CostAccumulator",
    # ----- FSM compiler -----
    "compile_fsm",
    # ----- Handler surface -----
    "compose",
    "HandlerTiming",
    "HandlerBuilder",
    "FSMHandler",
    "BaseHandler",
    "create_handler",
    # ----- Profiles -----
    "HarnessProfile",
    "ProviderProfile",
    "ProfileRegistry",
    "profile_registry",
    # ----- Root errors (full hierarchy at fsm_llm.errors) -----
    "FSMError",
    "LambdaError",
    # ----- Logging -----
    "setup_logging",
    # ============================================================
    # Dialog tier
    # ============================================================
    # FSM dialog core
    "ContextMergeStrategy",
    "FSMManager",
    "FSMDefinition",
    "FSMInstance",
    "FSMContext",
    "Conversation",
    "State",
    "Transition",
    "TransitionCondition",
    "TransitionOption",
    "TransitionEvaluation",
    # Classification & extraction
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
    "ClassificationExtractionConfig",
    "FieldExtractionConfig",
    # Pydantic request/response models
    "DataExtractionResponse",
    "ResponseGenerationRequest",
    "ResponseGenerationResponse",
    "FieldExtractionRequest",
    "FieldExtractionResponse",
    "LLMRequestType",
    "TransitionEvaluationResult",
    # Prompt builders
    "ClassificationPromptConfig",
    "DataExtractionPromptBuilder",
    "DataExtractionPromptConfig",
    "FieldExtractionPromptBuilder",
    "FieldExtractionPromptConfig",
    "ResponseGenerationPromptBuilder",
    "ResponsePromptConfig",
    "build_classification_json_schema",
    "build_classification_system_prompt",
    # Transition evaluation
    "TransitionEvaluator",
    "TransitionEvaluatorConfig",
    # Context & memory
    "ContextCompactor",
    "WorkingMemory",
    # Session persistence
    "FileSessionStore",
    "SessionState",
    "SessionStore",
    # Validation, visualization, loaders
    "load_fsm_definition",
    "load_fsm_from_file",
    "extract_json_from_text",
    "validate_json_structure",
    "get_fsm_summary",
    "evaluate_logic",
    "FSMValidator",
    "validate_fsm_from_file",
    "FSMValidationResult",
    "visualize_fsm_ascii",
    "visualize_fsm_from_file",
]


# ----------------------------------------------------------------------------
# Layer partition — used by tests/test_fsm_llm/test_layering.py to enforce
# that the public surface stays disjoint and covers __all__. Sub-namespace
# re-exports (fsm_llm.ast, fsm_llm.dsl, fsm_llm.combinators, fsm_llm.factories,
# fsm_llm.errors, fsm_llm.debug) live at L0 — they don't appear in the
# top-level __all__ so they're not partitioned here.
#
# These frozensets are PRIVATE (underscore-prefixed). Sole consumer is the
# layering audit test.
# ----------------------------------------------------------------------------

_LAYER_L4: frozenset[str] = frozenset(
    {
        "Program",
        "Result",
        "ExplainOutput",
        "ProgramModeError",
    }
)

_LAYER_L3: frozenset[str] = frozenset(
    {
        "compile_fsm",
    }
)

_LAYER_L2: frozenset[str] = frozenset(
    {
        # Composition surface
        "compose",
        # Handler API
        "FSMHandler",
        "BaseHandler",
        "HandlerTiming",
        "HandlerBuilder",
        "create_handler",
        # Profiles — apply-once data bundles
        "HarnessProfile",
        "ProviderProfile",
        "ProfileRegistry",
        "profile_registry",
    }
)

_LAYER_L1: frozenset[str] = frozenset(
    {
        # Planner
        "PlanInputs",
        "Plan",
        "plan",
        # Oracle + cost
        "Oracle",
        "LiteLLMOracle",
        "Executor",
        "CostAccumulator",
        # Root errors
        "LambdaError",
    }
)

# Hook for future deprecation cycles. Empty at 0.9.0.
_LAYER_LEGACY_DEPRECATED: frozenset[str] = frozenset()


# ----------------------------------------------------------------------------
# Module Metadata
# ----------------------------------------------------------------------------

__title__ = "fsm-llm"
__description__ = "Finite State Machines infused with Large Language Models"
__url__ = "https://github.com/NikolasMarkou/fsm_llm"
__author__ = "Nikolas Markou"
__email__ = "nikolasmarkou@gmail.com"
__license__ = "GPLv3"
__copyright__ = "Copyright 2025"
