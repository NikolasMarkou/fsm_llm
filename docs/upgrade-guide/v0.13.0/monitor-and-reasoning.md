# Monitor and reasoning

Covers FSM-LLM 0.12.0 to 0.13.0. Four public names of `fsm_llm.monitor` and `fsm_llm.reasoning` changed to follow the vocabulary of core: the monitor's errors now sit under `FSMError`, its manager has the lifecycle verbs of core, the reasoning root exception is `ReasoningError`, and the reasoning step budgets say that they count steps. None has an alias.

The 49 monitor route functions that moved out of `fsm_llm.monitor.server` in the same release are a separate change and are covered with the module paths.

## MonitorError is an FSMError

`fsm_llm.monitor.MonitorError` now subclasses `FSMError` (it was `Exception`), so `except FSMError` also catches `MonitorError` and its four subclasses (`MonitorInitializationError`, `MetricCollectionError`, `MonitorConnectionError`, `MonitorCapacityError`). The constructors are unchanged: a message, plus the optional `details` of `FSMError`. Code that relied on a `MonitorError` escaping an `except FSMError` clause must reorder its handlers: put the `except MonitorError` clause first.

**Before**

```python
try:
    run_monitor()
except FSMError:
    handle_core_error()  # a MonitorError was never caught here
except MonitorError:
    handle_monitor_error()
```

**After**

```python
from fsm_llm import FSMError
from fsm_llm.monitor import MonitorCapacityError, MonitorError

assert issubclass(MonitorError, FSMError)


def classify(error: Exception) -> str:
    try:
        raise error
    except MonitorError:  # first: a MonitorError is also an FSMError now
        return "monitor"
    except FSMError:
        return "core"


assert classify(MonitorCapacityError("too many instances")) == "monitor"
assert classify(FSMError("core failure")) == "core"
```

## InstanceManager close and end_instance

The `InstanceManager` lifecycle methods follow the vocabulary of core and the workflow engine: `shutdown` is now `close` (it stops the manager, like `API.close`) and `destroy_instance` is now `end_instance` (it finishes one managed instance of any kind, like `end_conversation`; `cancel_*` stays the abort verb). The REST route `DELETE /api/instances/{id}`, the event type `instance_destroyed` and the `OTELExporter.shutdown` method (which wraps the OpenTelemetry provider) are unchanged.

**Before**

```python
manager.destroy_instance(instance_id)
manager.shutdown()
```

**After**

```python
from fsm_llm.monitor import InstanceManager

manager = InstanceManager()
assert hasattr(manager, "end_instance") and not hasattr(manager, "destroy_instance")
manager.close()
assert not hasattr(manager, "shutdown")
```

## ReasoningEngineError is ReasoningError

`fsm_llm.reasoning.ReasoningEngineError` is renamed `ReasoningError`, the `<Package>Error` spelling of every other package root. `ReasoningExecutionError` and `ReasoningClassificationError` keep their names and now derive from `ReasoningError`. The export in `fsm_llm.reasoning.__all__` moves with it. An `except ReasoningEngineError` clause now fails with an `ImportError` at import time, before any problem can hide.

**Before**

```python
from fsm_llm.reasoning import ReasoningEngineError

try:
    solution, trace = engine.solve_problem(problem)
except ReasoningEngineError:
    ...
```

**After**

```python
import fsm_llm.reasoning as reasoning
from fsm_llm import FSMError
from fsm_llm.reasoning import (
    ReasoningClassificationError,
    ReasoningError,
    ReasoningExecutionError,
)

assert issubclass(ReasoningError, FSMError)
assert issubclass(ReasoningExecutionError, ReasoningError)
assert issubclass(ReasoningClassificationError, ReasoningError)
assert not hasattr(reasoning, "ReasoningEngineError")
```

## Reasoning step budgets end in STEPS

The reasoning step budgets of `fsm_llm.reasoning.constants.Defaults` end in `_STEPS`, because each one is a ceiling on core steps (`run_until_terminal(max_steps=...)` or the sub-FSM step counter). The values are unchanged. `MAX_SOLVE_STEPS` and `MAX_HYBRID_LOOPS` keep their names.

| Before | After |
| --- | --- |
| `MAX_TOTAL_ITERATIONS` | `MAX_TOTAL_STEPS` |
| `MAX_SUB_FSM_ITERATIONS` | `MAX_SUB_FSM_STEPS` |
| `MAX_CLASSIFICATION_ITERATIONS` | `MAX_CLASSIFICATION_STEPS` |

**Before**

```python
from fsm_llm.reasoning.constants import Defaults

limit = Defaults.MAX_TOTAL_ITERATIONS
```

**After**

```python
from fsm_llm.reasoning.constants import Defaults

assert Defaults.MAX_TOTAL_STEPS == 50
assert Defaults.MAX_SUB_FSM_STEPS == 30
assert Defaults.MAX_CLASSIFICATION_STEPS == 10
assert not hasattr(Defaults, "MAX_TOTAL_ITERATIONS")
```
