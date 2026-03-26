"""
Backward-compatibility shim.

All classification functionality has moved to ``fsm_llm``.
This package re-exports the public API for existing code that
imports from ``fsm_llm_classification``.
"""

import warnings

warnings.warn(
    "fsm_llm_classification is deprecated. Import from fsm_llm directly. "
    "Example: from fsm_llm import Classifier, ClassificationSchema",
    DeprecationWarning,
    stacklevel=2,
)

from fsm_llm.classification import (
    Classifier,
    HandlerFn,
    HierarchicalClassifier,
    IntentRouter,
)
from fsm_llm.definitions import (
    ClassificationError,
    ClassificationResponseError,
    ClassificationResult,
    ClassificationSchema,
    DomainSchema,
    HierarchicalResult,
    HierarchicalSchema,
    IntentDefinition,
    IntentScore,
    MultiClassificationResult,
    SchemaValidationError,
)
from fsm_llm.prompts import (
    ClassificationPromptConfig,
)
from fsm_llm.prompts import (
    build_classification_json_schema as build_json_schema,
)
from fsm_llm.prompts import (
    build_classification_system_prompt as build_system_prompt,
)

try:
    from fsm_llm.__version__ import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    # Version
    "__version__",
    # Schema
    "IntentDefinition",
    "ClassificationSchema",
    # Results
    "ClassificationResult",
    "IntentScore",
    "MultiClassificationResult",
    # Hierarchical
    "DomainSchema",
    "HierarchicalSchema",
    "HierarchicalResult",
    # Classifiers
    "Classifier",
    "HierarchicalClassifier",
    # Routing
    "IntentRouter",
    "HandlerFn",
    # Prompt utilities
    "ClassificationPromptConfig",
    "build_json_schema",
    "build_system_prompt",
    # Exceptions
    "ClassificationError",
    "SchemaValidationError",
    "ClassificationResponseError",
]
