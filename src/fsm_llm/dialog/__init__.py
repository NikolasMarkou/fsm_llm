from __future__ import annotations

"""
fsm_llm.dialog — FSM front-end (R4, plan v3 step 21).

Houses the FSM-dialog surface: API, FSMManager, MessagePipeline, Classifier
hierarchy, TransitionEvaluator, prompt builders, definitions, sessions, and
the FSM→λ compiler (`compile_fsm`, `compile_fsm_cached`). Moved from the
top-level `fsm_llm.*` namespace to this subpackage per `docs/lambda.md` §11.

# DECISION D-004 — silent shim policy:
# - 0.4.x: `from fsm_llm.dialog.api import API`, `from fsm_llm.dialog.fsm import FSMManager`,
#   `from fsm_llm.dialog.pipeline import MessagePipeline`, etc. all keep working via
#   sys.modules shims at `src/fsm_llm/{api,fsm,pipeline,...}.py`.
# - 0.5.0: emit DeprecationWarning at import time of each shim.
# - 0.6.0: remove the shims.
"""

from .api import API
from .classification import Classifier, HierarchicalClassifier, IntentRouter
from .compile_fsm import compile_fsm, compile_fsm_cached
from .definitions import (
    ClassificationExtractionConfig,
    ClassificationResult,
    ClassificationSchema,
    Conversation,
    FieldExtractionConfig,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    State,
    Transition,
)
from .fsm import FSMManager
from .prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
)
from .session import FileSessionStore, SessionState, SessionStore
from .transition_evaluator import TransitionEvaluator, TransitionEvaluatorConfig
from .turn import MessagePipeline

__all__ = [
    "API",
    "Classifier",
    "ClassificationExtractionConfig",
    "ClassificationResult",
    "ClassificationSchema",
    "Conversation",
    "DataExtractionPromptBuilder",
    "FieldExtractionConfig",
    "FieldExtractionPromptBuilder",
    "FileSessionStore",
    "FSMContext",
    "FSMDefinition",
    "FSMInstance",
    "FSMManager",
    "HierarchicalClassifier",
    "IntentRouter",
    "MessagePipeline",
    "ResponseGenerationPromptBuilder",
    "SessionState",
    "SessionStore",
    "State",
    "Transition",
    "TransitionEvaluator",
    "TransitionEvaluatorConfig",
    "compile_fsm",
    "compile_fsm_cached",
]
