# FSM-LLM Classification Extension

LLM-backed structured classification for mapping free-form user input to predefined intent classes with validated JSON output. Part of the `fsm-llm` package.

## Features

- **Single-Intent Classification**: Map user input to one of N predefined classes with confidence scoring and entity extraction.
- **Multi-Intent Classification**: Detect multiple intents in compound queries, ranked by confidence.
- **Hierarchical Classification**: Two-stage domain-then-intent classification for large class sets (>15 intents).
- **Intent Routing**: `IntentRouter` maps classified intents to handler functions with low-confidence clarification fallback.
- **Structured Output**: Uses `response_format` (JSON schema) when the LLM provider supports it; falls back to prompt-based enforcement.
- **Reasoning-First Schema**: The `reasoning` field precedes `intent` in the JSON schema to mitigate constrained-decoding probability distortion.

## Installation

```bash
pip install fsm-llm[classification]
```

## Quick Start

### Basic Classification

```python
from fsm_llm_classification import Classifier, ClassificationSchema, IntentDefinition

schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="order_status", description="User asks about an order"),
        IntentDefinition(name="product_info", description="User asks about a product"),
        IntentDefinition(name="payment_issue", description="User reports a payment problem"),
        IntentDefinition(name="return_request", description="User wants to return an item"),
        IntentDefinition(name="general_support", description="Anything else"),
    ],
    fallback_intent="general_support",
    confidence_threshold=0.6,
)

classifier = Classifier(schema, model="gpt-4o-mini")
result = classifier.classify("Where is my order #12345?")

print(result.intent)       # "order_status"
print(result.confidence)   # 0.95
print(result.entities)     # {"order_id": "12345"}
print(result.reasoning)    # "The user is asking about order tracking..."
```

### Intent Routing

```python
from fsm_llm_classification import IntentRouter

def handle_order_status(message: str, entities: dict) -> str:
    order_id = entities.get("order_id", "unknown")
    return f"Looking up order {order_id}..."

def handle_general(message: str, entities: dict) -> str:
    return "How can I help you today?"

router = IntentRouter(schema)
router.register("order_status", handle_order_status)
router.register("product_info", handle_product_info)
router.register("payment_issue", handle_payment_issue)
router.register("return_request", handle_return_request)
router.register("general_support", handle_general)

# Low-confidence inputs automatically trigger clarification
response = router.route(user_message, result)
```

### Multi-Intent Classification

```python
result = classifier.classify_multi("Where is order #123 and can I return order #456?")

for scored in result.intents:
    print(f"{scored.intent}: {scored.confidence:.2f} - {scored.entities}")
# order_status: 0.90 - {"order_id": "123"}
# return_request: 0.85 - {"order_id": "456"}
```

### Hierarchical Classification

For large class sets (>15 intents), use a two-stage classifier:

```python
from fsm_llm_classification import HierarchicalClassifier, HierarchicalSchema

domain_schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="billing", description="Billing and payment queries"),
        IntentDefinition(name="shipping", description="Shipping and delivery queries"),
        IntentDefinition(name="other", description="Everything else"),
    ],
    fallback_intent="other",
)

billing_schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="payment_failure", description="Payment failed"),
        IntentDefinition(name="refund_request", description="User wants a refund"),
        IntentDefinition(name="invoice_query", description="User asks about an invoice"),
        IntentDefinition(name="billing_other", description="Other billing query"),
    ],
    fallback_intent="billing_other",
)

shipping_schema = ClassificationSchema(
    intents=[
        IntentDefinition(name="tracking", description="Track a shipment"),
        IntentDefinition(name="delivery_issue", description="Delivery problem"),
        IntentDefinition(name="address_change", description="Change delivery address"),
        IntentDefinition(name="shipping_other", description="Other shipping query"),
    ],
    fallback_intent="shipping_other",
)

h_schema = HierarchicalSchema(
    domain_schema=domain_schema,
    intent_schemas={
        "billing": billing_schema,
        "shipping": shipping_schema,
    },
)

h_classifier = HierarchicalClassifier(h_schema, model="gpt-4o-mini")
result = h_classifier.classify("I want a refund for $50")
print(result.domain.intent)  # "billing"
print(result.intent.intent)  # "refund_request"
```

## Configuration

```python
from fsm_llm_classification import Classifier, ClassificationPromptConfig

config = ClassificationPromptConfig(
    include_reasoning=True,   # CoT reasoning before classification (recommended)
    include_entities=True,    # Extract entities from input
    multi_intent=False,       # Single-intent mode
    max_tokens=512,           # Token budget for classification
    temperature=0.0,          # Deterministic output
)

classifier = Classifier(schema, model="gpt-4o-mini", config=config)
```

## File Map

| File | Purpose |
|------|---------|
| `classifier.py` | **Classifier** (single/multi-intent) and **HierarchicalClassifier** (two-stage domain→intent) |
| `definitions.py` | Pydantic models: IntentDefinition, ClassificationSchema, ClassificationResult, MultiClassificationResult, HierarchicalSchema, HierarchicalResult |
| `prompts.py` | **ClassificationPromptConfig**, `build_system_prompt()`, `build_json_schema()` |
| `router.py` | **IntentRouter** — maps intents to handler functions with low-confidence fallback |
| `__version__.py` | Package version string |
| `__init__.py` | Public API exports — single `__all__` list |

## Schema Design Guidelines

- **Max ~15 intents** per classifier. Use hierarchical classification for more.
- **Always include a fallback** intent (set via `fallback_intent`).
- **Use `snake_case`** identifiers for easy handler mapping.
- **Keep classes mutually exclusive** and exhaustive.
- **Set `confidence_threshold`** (default 0.6) to trigger clarification on ambiguous inputs.

## Prompt Utilities

Build prompts and JSON schemas programmatically:

```python
from fsm_llm_classification import build_system_prompt, build_json_schema

# Get the system prompt for your schema
prompt = build_system_prompt(schema)

# Get the JSON schema (for constrained decoding or API-level structured output)
json_schema = build_json_schema(schema)
```

## API Reference

### Classifiers

| Class / Method | Description |
|-------|-------------|
| `Classifier(schema, model=..., config=...)` | Single-intent and multi-intent classification |
| `classifier.is_low_confidence(result)` | Check if a result falls below the schema's `confidence_threshold` |
| `HierarchicalClassifier(schema, model=...)` | Two-stage domain→intent classification |

### Schema Models

| Class | Description |
|-------|-------------|
| `IntentDefinition(name, description)` | Single intent definition |
| `ClassificationSchema(intents, fallback_intent, confidence_threshold)` | Schema for a set of intents |
| `DomainSchema` | Domain-level schema for hierarchical classification |
| `HierarchicalSchema(domain_schema, intent_schemas)` | Two-stage schema |

### Result Models

| Class | Key Fields |
|-------|------------|
| `ClassificationResult` | `intent`, `confidence`, `reasoning`, `entities` |
| `IntentScore` | `intent`, `confidence`, `entities` |
| `MultiClassificationResult` | `intents` (list of `IntentScore`) |
| `HierarchicalResult` | `domain` (ClassificationResult), `intent` (ClassificationResult) |

### Routing

| Class/Method | Description |
|-------------|-------------|
| `IntentRouter(schema)` | Create router for a schema |
| `router.register(intent, handler_fn)` | Register handler for intent |
| `router.route(message, result)` | Route single-intent result |
| `router.route_multi(message, result)` | Route multi-intent result |

### Exception Hierarchy

- `ClassificationError` → `SchemaValidationError`, `ClassificationResponseError`

## Examples

*   [`examples/classification/intent_routing/`](../../examples/classification/intent_routing/) — Basic intent classification with handler routing
*   [`examples/classification/smart_helpdesk/`](../../examples/classification/smart_helpdesk/) — Classification-driven FSM selection
*   [`examples/classification/classified_transitions/`](../../examples/classification/classified_transitions/) — Classification-aware FSM transition routing

The classification package also integrates with the agents package. See [`examples/agents/classified_dispatch/`](../../examples/agents/classified_dispatch/) and [`examples/agents/classified_tools/`](../../examples/agents/classified_tools/) for classification + agent patterns.

## Development

```bash
pytest tests/test_fsm_llm_classification/  # 52 tests across 5 test files
```

## License

GNU General Public License v3.0. See [LICENSE](../../LICENSE).
