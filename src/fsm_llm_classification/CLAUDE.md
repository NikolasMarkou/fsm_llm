# fsm_llm_classification -- Claude Code Instructions

## What This Package Does

LLM-backed structured intent classification. Maps free-form user input to predefined intent classes with validated JSON output. Three classification modes:

- **Single-intent** -- `Classifier.classify()` returns one `ClassificationResult` with intent, confidence, reasoning, entities
- **Multi-intent** -- `Classifier.classify_multi()` returns `MultiClassificationResult` with ranked `IntentScore` list (1-5 items)
- **Hierarchical (two-stage)** -- `HierarchicalClassifier.classify()` runs domain classification then intent classification within that domain

## File Map

| File | Purpose |
|------|---------|
| `classifier.py` | `Classifier` (single/multi-intent) and `HierarchicalClassifier` (two-stage domain->intent). Handles LLM calls, structured output, response parsing |
| `definitions.py` | Pydantic models: `IntentDefinition`, `ClassificationSchema`, `ClassificationResult`, `IntentScore`, `MultiClassificationResult`, `DomainSchema`, `HierarchicalSchema`, `HierarchicalResult`. Exception hierarchy |
| `prompts.py` | `ClassificationPromptConfig` (frozen dataclass), `build_system_prompt()`, `build_json_schema()`. Embeds JSON schema in prompt for prompt-only fallback |
| `router.py` | `IntentRouter` -- maps intent names to `HandlerFn` callables. Methods: `register()`, `register_many()`, `route()`, `route_multi()`, `validate()`. Low-confidence -> clarification handler |
| `__init__.py` | Public exports -- single `__all__` list with 18 symbols |
| `__version__.py` | Imports version from `fsm_llm.__version__` |

## Key Patterns

### Classification Flow (4 steps)
1. Define `ClassificationSchema` with `IntentDefinition` list + `fallback_intent` + `confidence_threshold`
2. Create `Classifier(schema, model=...)` or `HierarchicalClassifier(h_schema, model=...)`
3. Call `classifier.classify(text)` or `classifier.classify_multi(text)`
4. Route with `IntentRouter.route(message, result)` or `router.route_multi(message, multi_result)`

### Structured Output (response_format + fallback)
- `_call_llm()` checks `get_supported_openai_params(model)` for `response_format` support
- If supported: sends `response_format` with `json_schema` type
- If not: relies on prompt-embedded JSON schema for structure guidance
- Response parsing uses `extract_json_from_text()` (4-strategy fallback from `fsm_llm.utilities`)

### Ollama Handling
- `apply_ollama_params()` from `fsm_llm.ollama` disables thinking mode (`reasoning_effort="none"`) and forces `temperature=0`
- Thinking model fallback: if content is empty but `thinking` field has content, extracts JSON from thinking

### Confidence Thresholds
- `ClassificationSchema.confidence_threshold` defaults to 0.6
- `Classifier.is_low_confidence(result)` checks against schema threshold
- `ClassificationResult.is_low_confidence` property checks against class-level `DEFAULT_CONFIDENCE_THRESHOLD` (0.6)
- `IntentRouter.route()` redirects to clarification handler when below threshold
- `IntentRouter.route_multi()` skips intents below threshold

### JSON Schema Generation
- `reasoning` field precedes `intent` in schema ordering to mitigate constrained-decoding probability distortion
- Single-intent schema: `{reasoning, intent, confidence, entities}`
- Multi-intent schema: `{reasoning, intents: [{intent, confidence, entities}]}`
- `include_reasoning` and `include_entities` are configurable via `ClassificationPromptConfig`

## Dependencies on Core

- `fsm_llm.logging.logger` -- loguru logging
- `fsm_llm.ollama.apply_ollama_params` -- Ollama thinking-mode disable and temperature forcing
- `fsm_llm.constants.DEFAULT_LLM_MODEL` -- default model string
- `fsm_llm.utilities.extract_json_from_text` -- JSON extraction with 4-strategy fallback
- `fsm_llm.definitions.FSMError` -- base exception class
- `litellm.completion` and `litellm.get_supported_openai_params` -- LLM communication

## Exception Hierarchy

```
FSMError (from fsm_llm.definitions)
  ClassificationError          -- base for all classification errors
    SchemaValidationError      -- invalid schema (unused in code, available for consumers)
    ClassificationResponseError -- LLM returned empty or unparseable response
```

## Testing

```bash
pytest tests/test_fsm_llm_classification/ -v  # 52 tests
```

5 test files:

| File | Coverage |
|------|----------|
| `test_classifier.py` | Classifier and HierarchicalClassifier: LLM mocking, parsing, fallback behavior |
| `test_definitions.py` | Pydantic model validation, schema constraints, edge cases |
| `test_prompts.py` | Prompt generation, JSON schema building, config variations |
| `test_router.py` | IntentRouter: registration, routing, multi-intent routing, validation |
| `test_audit_fixes.py` | Regression tests for audit-identified issues |

## Gotchas

- **Max ~15 intents per schema** -- `ClassificationSchema` accepts more but quality degrades. Use `HierarchicalClassifier` for larger intent sets
- **`confidence_threshold` behavior** -- two separate mechanisms: `ClassificationResult.is_low_confidence` uses a class-level default (0.6), while `Classifier.is_low_confidence()` and `IntentRouter.route()` use the schema's threshold. Prefer the schema-aware methods
- **Ollama JSON schema handling** -- `apply_ollama_params()` modifies `call_params` in place, sets `json_schema` format and disables thinking. Some Ollama models return JSON in the thinking field instead of content
- **`HierarchicalClassifier` requires matching schemas** -- `HierarchicalSchema` validates that every non-fallback domain in `domain_schema` has a corresponding entry in `intent_schemas`
- **Unknown intents fall back silently** -- if the LLM returns an intent not in the schema, it is remapped to `fallback_intent` with a warning log (not an exception)
- **Multi-intent deduplication** -- if fallback remapping creates duplicate intent names, only the highest-confidence entry is kept
- **`IntentDefinition.name` validation** -- must be alphanumeric with underscores only (validated by Pydantic model_validator)
- **Prompt pre-building** -- `Classifier.__init__` pre-builds single-intent and multi-intent system prompts and JSON schemas. Schema changes after construction have no effect
- **`MultiClassificationResult.intents`** -- constrained to 1-5 items by Pydantic field validator
