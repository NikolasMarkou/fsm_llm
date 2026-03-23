# fsm_llm_classification — Classification Extension

## What This Package Does

LLM-backed structured intent classification. Maps free-form user input to predefined intent classes with validated JSON output. Supports single-intent, multi-intent, and hierarchical (two-stage) classification.

## File Map

| File | Purpose |
|------|---------|
| `classifier.py` | **Classifier** (single/multi-intent) and **HierarchicalClassifier** (two-stage domain→intent) |
| `definitions.py` | Pydantic models: IntentDefinition, ClassificationSchema, ClassificationResult, MultiClassificationResult, HierarchicalSchema, HierarchicalResult |
| `prompts.py` | **ClassificationPromptConfig**, `build_system_prompt()`, `build_json_schema()` |
| `router.py` | **IntentRouter** — maps intents to handler functions with low-confidence fallback |
| `__init__.py` | Public exports — single `__all__` list |
| `__version__.py` | Package version string |

## Key Patterns

### Classification Flow
1. Define `ClassificationSchema` with `IntentDefinition` list + `fallback_intent` + `confidence_threshold`
2. Create `Classifier(schema, model=...)` or `HierarchicalClassifier(h_schema, model=...)`
3. Call `classifier.classify(text)` → `ClassificationResult` or `classifier.classify_multi(text)` → `MultiClassificationResult`
4. Route with `IntentRouter.route(message, result)`

### Structured Output
- Uses `response_format` (JSON schema) when LLM provider supports it
- Falls back to prompt-based enforcement
- `reasoning` field precedes `intent` in schema to mitigate constrained-decoding distortion

### Classifier Internals (classifier.py)
- `_call_llm()` handles LLM communication with structured output support
- Ollama models: uses `fsm_llm.ollama` helpers to disable thinking (`reasoning_effort="none"`) and force `temperature=0`
- Support for thinking models (extracts content from thinking field as fallback)
- Uses `fsm_llm.constants.DEFAULT_LLM_MODEL` and `fsm_llm.logging.logger`

### Exception Hierarchy
`ClassificationError` → `SchemaValidationError`, `ClassificationResponseError`

## Dependencies on Core
- `fsm_llm.logging.logger` — logging
- `fsm_llm.ollama` — Ollama thinking-mode disable
- `fsm_llm.constants.DEFAULT_LLM_MODEL` — default model
- `litellm` — LLM completions

## Testing
```bash
pytest tests/test_fsm_llm_classification/  # 52 unit tests
```
5 test files: test_classifier.py, test_definitions.py, test_prompts.py, test_router.py

## Gotchas
- Max ~15 intents per classifier — use HierarchicalClassifier for more
- Always set `fallback_intent` in schema
- `confidence_threshold` defaults to 0.6
- Ollama models use `json_schema` structured output with thinking disabled via `reasoning_effort="none"`
