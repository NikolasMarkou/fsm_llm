# fsm_llm_classification -- Deprecation Shim

## Status

**This package is now a thin deprecation shim.** All classification code has been absorbed into the `fsm_llm` core package. Importing from `fsm_llm_classification` still works but emits a `DeprecationWarning`.

## Where the Code Moved

| Former location | New location |
|----------------|-------------|
| `fsm_llm_classification.classifier` (`Classifier`, `HierarchicalClassifier`, `IntentRouter`, `HandlerFn`) | `fsm_llm.classification` |
| `fsm_llm_classification.definitions` (`ClassificationSchema`, `IntentDefinition`, `ClassificationResult`, `IntentScore`, `MultiClassificationResult`, `DomainSchema`, `HierarchicalSchema`, `HierarchicalResult`) | `fsm_llm.definitions` |
| `fsm_llm_classification.prompts` (`ClassificationPromptConfig`, `build_system_prompt`, `build_json_schema`) | `fsm_llm.prompts` (exported as `build_classification_system_prompt`, `build_classification_json_schema`) |
| `fsm_llm_classification.definitions` (exceptions: `ClassificationError`, `SchemaValidationError`, `ClassificationResponseError`) | `fsm_llm.definitions` |

## Preferred Imports

```python
# New (preferred)
from fsm_llm import Classifier, ClassificationSchema, IntentRouter
from fsm_llm.classification import Classifier, HierarchicalClassifier

# Old (deprecated, emits DeprecationWarning)
from fsm_llm_classification import Classifier, ClassificationSchema
```

## Testing

```bash
pytest tests/test_fsm_llm_classification/ -v  # 52 tests
```

Tests still exercise the shim to ensure backward compatibility, but all logic is tested in `tests/test_fsm_llm/` as well.
