"""
FSM-LLM Classification Extension
=================================

LLM-backed structured classification for mapping free-form user input
to predefined intent classes with validated JSON output.

Basic Usage::

    from fsm_llm_classification import Classifier, ClassificationSchema, IntentDefinition

    schema = ClassificationSchema(
        intents=[
            IntentDefinition(name="order_status", description="User asks about an order"),
            IntentDefinition(name="product_info", description="User asks about a product"),
            IntentDefinition(name="general_support", description="Anything else"),
        ],
        fallback_intent="general_support",
    )

    classifier = Classifier(schema, model="gpt-4o-mini")
    result = classifier.classify("Where is my order #12345?")
    print(result.intent)       # "order_status"
    print(result.confidence)   # 0.95
    print(result.entities)     # {"order_id": "12345"}

Handler Routing::

    from fsm_llm_classification import IntentRouter

    router = IntentRouter(schema)
    router.register("order_status", handle_order_status)
    router.register("general_support", handle_general)

    response = router.route(user_message, result)
"""

from .__version__ import __version__
from .classifier import (
    Classifier,
    HierarchicalClassifier,
)
from .definitions import (
    # Exceptions
    ClassificationError,
    ClassificationResponseError,
    # Single-intent results
    ClassificationResult,
    ClassificationSchema,
    # Hierarchical classification
    DomainSchema,
    HierarchicalResult,
    HierarchicalSchema,
    # Schema building
    IntentDefinition,
    IntentScore,
    # Multi-intent results
    MultiClassificationResult,
    SchemaValidationError,
)
from .prompts import (
    ClassificationPromptConfig,
    build_json_schema,
    build_system_prompt,
)
from .router import (
    HandlerFn,
    IntentRouter,
)

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
