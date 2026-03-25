# fsm_llm_classification

LLM-backed structured intent classification extension for FSM-LLM. Maps free-form user input to predefined intent classes with validated JSON output, confidence scoring, entity extraction, and handler routing.

Part of the [FSM-LLM](https://github.com/NikolasMarkou/fsm_llm) framework (v0.3.0).

## Features

- **Single-intent classification** -- classify input into exactly one intent with confidence score
- **Multi-intent classification** -- detect multiple intents in compound queries, ranked by confidence
- **Hierarchical (two-stage) classification** -- domain-level classification followed by intent-level classification for large intent sets
- **Structured JSON output** -- uses `response_format` when the provider supports it, with prompt-based fallback
- **Confidence thresholds** -- configurable per-schema threshold with automatic low-confidence detection
- **Entity extraction** -- extracts relevant entities (order IDs, product names, etc.) alongside classification
- **Intent routing** -- map intents to handler functions with fallback and clarification support
- **Ollama support** -- automatic thinking-mode disable and temperature forcing for local models

## Installation

```bash
pip install fsm-llm[classification]
```

## Quick Start

Define a schema with intent definitions, then classify user input:

```python
from fsm_llm_classification import Classifier, ClassificationSchema, IntentDefinition

schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="order_status", description="User asks about an order"),
        IntentDefinition(name="product_info", description="User asks about a product"),
        IntentDefinition(name="general_support", description="Anything else"),
    ],
    fallback_intent="general_support",
    confidence_threshold=0.6,
)

classifier = Classifier(schema, model="gpt-4o-mini")
result = classifier.classify("Where is my order #12345?")

print(result.intent)       # "order_status"
print(result.confidence)   # 0.95
print(result.reasoning)    # "The user is asking about the status of order #12345..."
print(result.entities)     # {"order_id": "12345"}
```

For messages that may contain multiple intents:

```python
multi_result = classifier.classify_multi("Cancel order #123 and show me your laptops")

for scored in multi_result.intents:
    print(f"{scored.intent}: {scored.confidence:.2f}")
```

## Hierarchical Classification

When the total number of intents exceeds ~15, use `HierarchicalClassifier` to split classification into two stages: domain classification followed by intent classification within that domain.

```python
from fsm_llm_classification import (
    HierarchicalClassifier,
    HierarchicalSchema,
    ClassificationSchema,
    IntentDefinition,
)

# Stage 1: classify the domain
domain_schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="sales", description="Sales and purchasing"),
        IntentDefinition(name="support", description="Technical support"),
        IntentDefinition(name="general", description="General inquiries"),
    ],
    fallback_intent="general",
)

# Stage 2: per-domain intent schemas
sales_schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="pricing", description="Pricing questions"),
        IntentDefinition(name="availability", description="Stock availability"),
        IntentDefinition(name="sales_other", description="Other sales queries"),
    ],
    fallback_intent="sales_other",
)

support_schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="bug_report", description="Report a bug"),
        IntentDefinition(name="setup_help", description="Installation or setup help"),
        IntentDefinition(name="support_other", description="Other support queries"),
    ],
    fallback_intent="support_other",
)

h_schema = HierarchicalSchema(
    domain_schema=domain_schema,
    intent_schemas={
        "sales": sales_schema,
        "support": support_schema,
    },
)

h_classifier = HierarchicalClassifier(h_schema, model="gpt-4o-mini")
result = h_classifier.classify("How much does the Pro plan cost?")

print(result.domain.intent)  # "sales"
print(result.intent.intent)  # "pricing"
```

## Intent Router

Map classified intents to handler functions using `IntentRouter`:

```python
from fsm_llm_classification import IntentRouter

def handle_order_status(message: str, entities: dict[str, str]):
    order_id = entities.get("order_id", "unknown")
    return f"Looking up order {order_id}..."

def handle_general(message: str, entities: dict[str, str]):
    return "How can I help you?"

router = IntentRouter(schema)
router.register("order_status", handle_order_status)
router.register("general_support", handle_general)

# Route based on classification result
result = classifier.classify("Where is my order #12345?")
response = router.route("Where is my order #12345?", result)

# Validate all intents have handlers
missing = router.validate()  # Returns list of unregistered intent names
```

The router automatically handles low-confidence results by calling a clarification handler (customizable via the `clarification_handler` parameter). For multi-intent results, use `router.route_multi()` which routes each detected intent and skips those below the confidence threshold.

## API Reference

### Classifiers

| Class | Description |
|-------|-------------|
| `Classifier` | Single-intent and multi-intent classification. Methods: `classify(text)`, `classify_multi(text)`, `is_low_confidence(result)` |
| `HierarchicalClassifier` | Two-stage classification (domain then intent). Method: `classify(text)` |

### Schema Models

| Class | Description |
|-------|-------------|
| `IntentDefinition` | Single intent: `name` (snake_case alphanumeric) + `description` |
| `ClassificationSchema` | Intent list (2-15 recommended) + `fallback_intent` + `confidence_threshold` (default 0.6) |
| `HierarchicalSchema` | `domain_schema` + `intent_schemas` dict mapping domain names to `ClassificationSchema` |
| `DomainSchema` | Maps a domain name to its intent sub-schema |

### Result Models

| Class | Description |
|-------|-------------|
| `ClassificationResult` | Single-intent result: `intent`, `confidence`, `reasoning`, `entities`. Property: `is_low_confidence` |
| `MultiClassificationResult` | Multi-intent result: `reasoning` + ranked `intents` list of `IntentScore` (1-5 items). Property: `primary` |
| `IntentScore` | Scored intent within multi-intent result: `intent`, `confidence`, `entities` |
| `HierarchicalResult` | Two-stage result: `domain` + `intent` (both `ClassificationResult`) |

### Routing

| Class | Description |
|-------|-------------|
| `IntentRouter` | Intent-to-handler registry. Methods: `register()`, `register_many()`, `route()`, `route_multi()`, `validate()` |
| `HandlerFn` | Type alias: `Callable[[str, dict[str, str]], Any]` |

### Prompt Utilities

| Symbol | Description |
|--------|-------------|
| `ClassificationPromptConfig` | Frozen dataclass: `include_reasoning`, `include_entities`, `temperature`, `max_tokens`, `multi_intent`, `max_intents` |
| `build_system_prompt()` | Build system prompt from schema + config |
| `build_json_schema()` | Build JSON Schema dict for structured output |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `ClassificationError` | Base exception (inherits from `FSMError`) |
| `SchemaValidationError` | Invalid classification schema |
| `ClassificationResponseError` | LLM returned an unparseable response |

## File Map

| File | Purpose |
|------|---------|
| `classifier.py` | `Classifier` (single/multi-intent) and `HierarchicalClassifier` (two-stage) |
| `definitions.py` | Pydantic models, result types, and exception hierarchy |
| `prompts.py` | `ClassificationPromptConfig`, `build_system_prompt()`, `build_json_schema()` |
| `router.py` | `IntentRouter` with low-confidence fallback and multi-intent routing |
| `__init__.py` | Public exports (18 symbols in `__all__`) |
| `__version__.py` | Package version (synced from `fsm_llm.__version__`) |

## Examples

Three classification examples in `examples/classification/`:

| Example | Description |
|---------|-------------|
| `intent_routing` | Basic classifier with intent router |
| `smart_helpdesk` | Multi-domain helpdesk with hierarchical classification |
| `classified_transitions` | Classification-driven FSM state transitions |

Run any example with:

```bash
python examples/classification/<example_name>/run.py
```

## Integration

The classification package integrates with the agents package for advanced patterns:

- **classified_dispatch** (`examples/agents/classified_dispatch/`) -- classify user intent, then dispatch to specialized agents
- **classified_tools** (`examples/agents/classified_tools/`) -- use classification to select tools for a ReAct agent

## Development

```bash
# Run all 52 classification tests (5 test files)
pytest tests/test_fsm_llm_classification/ -v

# Lint and format
ruff check src/fsm_llm_classification/
ruff format src/fsm_llm_classification/
```

Test files: `test_classifier.py`, `test_definitions.py`, `test_prompts.py`, `test_router.py`, `test_audit_fixes.py`.
