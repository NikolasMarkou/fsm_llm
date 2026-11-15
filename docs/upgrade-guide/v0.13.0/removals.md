# Removals and small API changes

Covers FSM-LLM 0.12.0 to 0.13.0. This page collects the removals and the small signature changes of this release that fit no other guide: dead parameters, unused helpers and constants, and a few steps and handlers that nothing called. None has an alias and none has a raising stub: the old name fails at once with an `ImportError`, an `AttributeError` or a `TypeError`. In most sections the honest advice is that nothing in your code needs to change, because nothing in the repository called the removed name. One section (the planner harness) removes a whole package, and one (workflow steps) changes behaviour for a caller that drives `execute()` itself.

The sections follow the order of the CHANGELOG entries. A search of your code for the old name finds every place to edit.

## Six orchestrator and ADaPT prompt functions are removed

Six prompt functions are removed from `fsm_llm.agents.prompts`: `build_orchestrate_response_instructions`, `build_delegate_response_instructions`, `build_collect_response_instructions`, `build_attempt_response_instructions`, `build_assess_response_instructions` and `build_decompose_response_instructions`. They only fed the Orchestrator and ADaPT intermediate states, which are now silent (a state with no `response_instructions` makes no LLM call). There is nothing to replace them with: delete the call.

**Before**

```python
from fsm_llm.agents.prompts import build_delegate_response_instructions

text = build_delegate_response_instructions()
```

**After**

```python
import fsm_llm.agents.prompts as prompts

removed = (
    "build_orchestrate_response_instructions",
    "build_delegate_response_instructions",
    "build_collect_response_instructions",
    "build_attempt_response_instructions",
    "build_assess_response_instructions",
    "build_decompose_response_instructions",
)
assert not any(hasattr(prompts, name) for name in removed)
assert hasattr(prompts, "build_orchestrator_synthesize_response_instructions")
```

## build_adapt_field_instructions takes no parameters

`build_adapt_field_instructions(registry=None, task_description=None)` loses both parameters and takes none: ADaPT advertises no tool, so neither was read for the typed field instructions. A caller that passed either now gets a `TypeError`.

**Before**

```python
from fsm_llm.agents.prompts import build_adapt_field_instructions

fields = build_adapt_field_instructions(registry=registry, task_description="plan")
```

**After**

```python
from fsm_llm.agents.prompts import build_adapt_field_instructions

fields = build_adapt_field_instructions()
assert isinstance(fields, dict)

try:
    build_adapt_field_instructions(task_description="plan")
except TypeError:
    pass
else:
    raise AssertionError("the old parameter was accepted")
```

## AgentGraph.run takes no config

`AgentGraph.run(task, initial_context=None)` no longer takes `config`, with no raising stub: the argument was never read (`config=AgentConfig(max_iterations=1)` changed nothing), and a graph-level config has no meaning for member agents that are already built. Passing it is now a `TypeError`. Configure each agent when you build it.

**Before**

```python
result = graph.run("task", config=AgentConfig(max_iterations=1))
```

**After**

```python
import inspect

from fsm_llm.agents import AgentGraph

assert list(inspect.signature(AgentGraph.run).parameters) == [
    "self",
    "task",
    "initial_context",
]
```

**Sketch**

```python
from fsm_llm.agents import AgentConfig, ReactAgent

# The limit belongs to the member agent, where it is read.
member = ReactAgent(tools=registry, config=AgentConfig(max_iterations=1))
```

## SwarmAgent takes no api keywords

`SwarmAgent(..., **api_kwargs)` is removed. A swarm builds no API (its members own theirs), so `llm_interface=`, `seed=` and even a mistyped keyword were accepted and ignored. The signature is `SwarmAgent(agents, entry_agent, max_handoffs=10, memory=None, config=None)`. Any other keyword, also through `create_agent("swarm", ..., **kwargs)`, is a `TypeError`.

**Before**

```python
swarm = SwarmAgent(agents, "triage", llm_interface=llm, seed=7)
```

**After**

```python
import inspect

from fsm_llm.agents import SwarmAgent

parameters = list(inspect.signature(SwarmAgent.__init__).parameters)
assert parameters == [
    "self",
    "agents",
    "entry_agent",
    "max_handoffs",
    "memory",
    "config",
]
```

## build_adapt_fsm loses its registry parameter

`build_adapt_fsm(registry=None, task_description="", max_depth=3, context_keys=())` loses its first parameter `registry`: ADaPT runs no tool and the parameter had no effect on the FSM. The signature is `build_adapt_fsm(task_description="", max_depth=3, context_keys=())`. A positional caller that passed the registry or `None` first (`build_adapt_fsm(None, "task", 2)`) must drop it, and `registry=` is a `TypeError`. `ADaPTAgent(tools=...)` is unchanged. The FSM output is byte-identical, so no golden hash moves.

**Before**

```python
fsm = build_adapt_fsm(None, "task", 2)
```

**After**

```python
from fsm_llm.agents.adapt import build_adapt_fsm

fsm = build_adapt_fsm("task", 2)
assert isinstance(fsm, dict)

try:
    build_adapt_fsm(registry=None)
except TypeError:
    pass
else:
    raise AssertionError("the old parameter was accepted")
```

## API active_conversations and the eval exit constants are removed

`API.active_conversations` (a dict whose keys were always those of `conversation_stacks`) is removed: `list_active_conversations()` is the public read, `conversation_stacks` the registry. `fsm_llm.eval.constants.EXIT_OK` and `EXIT_ERROR` are removed: they repeated core's `CLI_EXIT_OK` and `CLI_EXIT_FAILURE` (`EXIT_BELOW_THRESHOLD = 2` stays eval's own).

**Before**

```python
ids = list(api.active_conversations)
from fsm_llm.eval.constants import EXIT_ERROR, EXIT_OK
```

**After**

```python
import fsm_llm.eval.constants as eval_constants
from fsm_llm import API
from fsm_llm.constants import CLI_EXIT_FAILURE, CLI_EXIT_OK

definition = {
    "name": "Bot",
    "description": "A one-state bot",
    "initial_state": "hello",
    "states": {
        "hello": {
            "id": "hello",
            "description": "Greet",
            "purpose": "Say hello",
            "response_instructions": "Say hello",
        }
    },
}
api = API.from_definition(definition, model="ollama_chat/example")
assert api.list_active_conversations() == []
assert not hasattr(api, "active_conversations")

assert (CLI_EXIT_OK, CLI_EXIT_FAILURE) == (0, 1)
assert not hasattr(eval_constants, "EXIT_OK")
assert not hasattr(eval_constants, "EXIT_ERROR")
assert eval_constants.EXIT_BELOW_THRESHOLD == 2
```

## PAUSING_STEP_TYPES is removed

`fsm_llm.workflows.constants.PAUSING_STEP_TYPES` (a set of class names as strings) is removed: it was redundant with the `isinstance(inner, (WaitEventStep, TimerStep))` check in `WorkflowDefinition._is_pausing_step`, which is now the only rule, and the string set would have gone stale on the step class renames. A step type that pauses is added to that `isinstance` check.

**Before**

```python
from fsm_llm.workflows.constants import PAUSING_STEP_TYPES

pauses = type(step).__name__ in PAUSING_STEP_TYPES
```

**After**

```python
import fsm_llm.workflows.constants as constants
from fsm_llm.workflows import TimerStep, WaitEventStep

assert not hasattr(constants, "PAUSING_STEP_TYPES")
pausing = (WaitEventStep, TimerStep)
```

## The reasoning load_fsm_definition is renamed get_reasoning_fsm

`fsm_llm.reasoning.utilities.load_fsm_definition` is renamed `get_reasoning_fsm` (no alias). It shared its name with the core export `fsm_llm.load_fsm_definition` and returns a different type (a deep copy of a dict from `ALL_REASONING_FSMS`). It was never in `fsm_llm.reasoning.__all__`.

**Before**

```python
from fsm_llm.reasoning.utilities import load_fsm_definition

fsm = load_fsm_definition("orchestrator")
```

**After**

```python
import fsm_llm.reasoning.utilities as utilities
from fsm_llm.reasoning.utilities import get_reasoning_fsm

assert not hasattr(utilities, "load_fsm_definition")
fsm = get_reasoning_fsm("orchestrator")
assert isinstance(fsm, dict)
assert fsm is not get_reasoning_fsm("orchestrator")  # a copy each time
```

## The planner harness is removed

The iterative-planner harness is removed, with no alias, no shim and no deprecation period. If you never imported `fsm_llm.harness`, installed the `harness` extra or ran `fsm-llm-harness`, nothing changes for you. Gone with it:

- the package `fsm_llm.harness` (and its `README.md` and `AGENTS.md`), so `from fsm_llm.harness import ...` and `python -m fsm_llm.harness` fail; `HarnessAgent`, `HarnessAgentBuilder` and the `HarnessError` tree (`HarnessError` was a subclass of `FSMError`) no longer exist;
- the console script `fsm-llm-harness` (reinstall with `pip install -e .` to drop the stale script);
- the extra `harness`, and its place in `all`: `pip install "llm-fsm[harness]"` now warns that the extra does not exist;
- the live-test gate `FSM_LLM_HARNESS_LIVE`, and the test suite `tests/test_fsm_llm_harness/`;
- the CLI and the live half of `scripts/harness_bench.py` (`register`, `run`, `report`, `probe-seed`). Its stdlib helpers moved to `scripts/bench_common.py`, which has no CLI; the tests of the moved helpers are `tests/test_bench_common.py` (was `tests/test_harness_bench.py`);
- the harness blocks under `scripts/bench_data/` (`l4-execute-write`, `l6-e2e`, `l7-explore-coldstart`, `l8-explore-loop`, `seed-probe`) are kept as frozen history: the code that produced them is gone, so they cannot be re-run, and they are not edited.

**Before**

```python
from fsm_llm.harness import HarnessAgent, HarnessError
```

```bash
pip install "fsm-llm[harness]"
fsm-llm-harness --help
```

**After**

```python
import importlib.util

assert importlib.util.find_spec("fsm_llm.harness") is None
```

## MIN_ANSWER_LENGTH is removed

`Defaults.MIN_ANSWER_LENGTH` (`fsm_llm.agents.constants`, value 5) is removed, with no alias: nothing reads it any more (the short-answer entry under Fixed in the CHANGELOG explains why). `EvaluatorOptimizerAgent._extract_answer` (private) is gone too; the base chain gives the same result. If you used the constant as a threshold of your own, define it in your code.

**Before**

```python
from fsm_llm.agents.constants import Defaults

too_short = len(answer) < Defaults.MIN_ANSWER_LENGTH
```

**After**

```python
from fsm_llm.agents.constants import Defaults

assert not hasattr(Defaults, "MIN_ANSWER_LENGTH")
MIN_ANSWER_LENGTH = 5  # yours now
too_short = len("ok") < MIN_ANSWER_LENGTH
assert too_short
```

## Unused agent handler methods and prompt helpers are removed

`ThinkTurnLimiter.reset` and `AgentHandlers.reset` (`fsm_llm.agents.handlers`) are removed: nothing called them, because every run builds a fresh instance. `fsm_llm.agents.base.with_instructions` and `prompt_overflow_error` are now private (`_with_instructions`, `_prompt_overflow_error`; `_create_api` is their only caller). None of the four was exported from `fsm_llm.agents`. Build a new handler object instead of resetting one.

**Before**

```python
limiter.reset()
handlers.reset()
```

**After**

```python
import fsm_llm.agents.base as base
from fsm_llm.agents.handlers import AgentHandlers, ThinkTurnLimiter

assert not hasattr(ThinkTurnLimiter, "reset")
assert not hasattr(AgentHandlers, "reset")
assert not hasattr(base, "with_instructions")
assert not hasattr(base, "prompt_overflow_error")
```

## Unused monitor methods are removed

These members of `fsm_llm.monitor` are removed, with no alias, because nothing in `src/`, the SPA or the docs called them: `EventCollector.get_events_since`, `get_logs_since`, `get_events_by_conversation`, `handler_name` and `handler_priority`; `InstanceManager.find_instance_for_conversation`; and `OTELExporter.generate_trace_id` and `generate_span_id`. The two `*_since` forms were the pre-cursor streaming API and drop the middle of a burst: use `events_after` and `logs_after`.

`InstanceManager.launch_workflow` loses its `initial_context` parameter, which was accepted and ignored: pass the context to `start_workflow_instance` instead.

**Before**

```python
events = collector.get_events_since(timestamp)
logs = collector.get_logs_since(timestamp)
manager.launch_workflow(preset_id="demo", initial_context={"x": 1})
```

**After**

```python
import inspect

from fsm_llm.monitor import EventCollector, InstanceManager

collector = EventCollector()
events, cursor = collector.events_after(0)
logs, cursor = collector.logs_after(0)
assert events == [] and logs == [] and cursor == 0

for name in ("get_events_since", "get_logs_since", "get_events_by_conversation"):
    assert not hasattr(EventCollector, name)
assert not hasattr(InstanceManager, "find_instance_for_conversation")
assert "initial_context" not in inspect.signature(InstanceManager.launch_workflow).parameters
```

## WorkflowValidator and unused resolver methods are removed

`fsm_llm.workflows.WorkflowValidator` is removed (also from `fsm_llm.workflows.__all__`), with no alias: nothing called it, because it only wrapped the definition's own check. Use `WorkflowDefinition.validate_structure()` (the 0.12.0 spelling `validate()` is renamed in this release, see the workflows vocabulary guide), which raises `WorkflowValidationError` with `validation_errors`.

`DependencyResolver` (still exported) loses `add_dependency`, `get_dependencies`, `get_dependents`, `get_all_steps` and `dependency_count`, which nothing called; `add_step`, `resolve`, `has_cycles`, `step_count`, `clear` and `from_dict` stay. `Timer.is_expired` (`fsm_llm.workflows.Timer`) is removed: only `EventListener.is_expired` was ever called.

**Before**

```python
from fsm_llm.workflows import WorkflowValidator

errors = WorkflowValidator.validate_workflow(workflow)
```

**After**

```python
import fsm_llm.workflows as workflows
from fsm_llm.workflows import (
    DependencyResolver,
    Timer,
    WorkflowValidationError,
    auto_step,
    create_workflow,
)

workflow = create_workflow("w", "W").with_initial_step(
    auto_step("a", "A", next_step_id="missing")
)
try:
    workflow.validate_structure()
except WorkflowValidationError as error:
    assert error.validation_errors

assert not hasattr(workflows, "WorkflowValidator")
assert "WorkflowValidator" not in workflows.__all__
for name in ("add_dependency", "get_dependencies", "get_dependents", "get_all_steps"):
    assert not hasattr(DependencyResolver, name)
assert not hasattr(Timer, "is_expired")
```

## Auto and condition steps return a failure result

`AutoTransitionStep` (now `AutoStep`, see the workflows vocabulary guide) and `ConditionStep` no longer RAISE `WorkflowStepError` from `execute()` when the action or condition fails and no error step is set. They return a failure result with no next step, like `APIStep`, `LLMStep` and the others, and the engine fails the instance from it. A caller that drove `execute()` itself must read `result.success` and `result.error` instead of catching the exception. An engine user sees the same end state: the engine fails the instance from the failure result.

`LLMStep(llm_interface=...)` is validated at construction: an object with neither a callable `generate` nor a callable `generate_response` raises `ValueError` (pydantic `ValidationError`) instead of failing the instance at run time.

**Before**

```python
try:
    result = await step.execute(context)
except WorkflowStepError:
    ...
```

**After**

```python
import asyncio

from fsm_llm.workflows import LLMStep, auto_step, condition_step


def explode(context):
    raise RuntimeError("bad action")


step = auto_step("a", "A", next_step_id="b", action=explode)
result = asyncio.run(step.execute({}))
assert result.success is False and "bad action" in str(result.error)
assert result.next_step_id is None

check = condition_step(
    "c", "C", condition=lambda ctx: 1 / 0, true_step_id="t", false_step_id="f"
)
result = asyncio.run(check.execute({}))
assert result.success is False and result.next_step_id is None

try:
    LLMStep(step_id="l", name="L", prompt_template="x", llm_interface=object())
except ValueError:
    pass
else:
    raise AssertionError("an object with no generate method was accepted")
```

## session_json_default and two unused names are removed

`session_json_default` (in 0.12.0 `fsm_llm.session`, now the module `fsm_llm.engine.session`) is removed, with no alias: use `redacting_json_default` from `fsm_llm.context.filtering` (the `fsm-llm` command's dump now uses it, so a secret-looking object prints as `<redacted:Type>`). `fsm_llm.logging.is_library_logging_enabled` is removed (nothing read it). `JsonLogicExpression` (an alias of `Any`, in 0.12.0 `fsm_llm.expressions`, now the package `fsm_llm.core.expressions`) is removed.

**Before**

```python
import json

from fsm_llm.session import session_json_default
from fsm_llm.expressions import JsonLogicExpression

text = json.dumps(context, default=session_json_default)
```

**After**

```python
import json

import fsm_llm.core.expressions as expressions
import fsm_llm.engine.session as session
import fsm_llm.logging as logging
from fsm_llm.context.filtering import redacting_json_default

text = json.dumps({"when": object()}, default=redacting_json_default)
assert "<redacted:object>" in text
assert not hasattr(session, "session_json_default")
assert not hasattr(logging, "is_library_logging_enabled")
assert not hasattr(expressions, "JsonLogicExpression")
```

## Agents constants and meta prompt helpers are removed or renamed

In `fsm_llm.agents`:

- `MetaDefaults.MODEL` is removed (use `resolve_model`, see the default model guide);
- `HandlerNames.ADAPT_ASSESSOR` is renamed `ADAPT_DECOMPOSE_TRACKER` (the handler name string `ADaPTAssessor` is unchanged);
- `HandlerNames.END_CONVERSATION` is removed together with the marker handler that wrote `_agent_completed` and `_agent_type` (nothing read them);
- `parallel_react.TOOL_CALLS_KEY` is removed in favour of `ContextKeys.TOOL_CALLS`;
- `meta_prompts.build_followup_message` and `build_output_message` are removed, and `meta_prompts.build_collect_response_instructions` is renamed `build_meta_collect_response_instructions` (the orchestrator one in `agents.prompts` keeps the old name).

**Before**

```python
from fsm_llm.agents.constants import HandlerNames, MetaDefaults
from fsm_llm.agents.meta_prompts import build_collect_response_instructions

name = HandlerNames.ADAPT_ASSESSOR
model = MetaDefaults.MODEL
```

**After**

```python
import fsm_llm.agents.meta_prompts as meta_prompts
import fsm_llm.agents.parallel_react as parallel_react
from fsm_llm.agents.constants import ContextKeys, HandlerNames, MetaDefaults

assert HandlerNames.ADAPT_DECOMPOSE_TRACKER
assert not hasattr(HandlerNames, "ADAPT_ASSESSOR")
assert not hasattr(HandlerNames, "END_CONVERSATION")
assert not hasattr(MetaDefaults, "MODEL")
assert not hasattr(parallel_react, "TOOL_CALLS_KEY")
assert ContextKeys.TOOL_CALLS == "tool_calls"
assert hasattr(meta_prompts, "build_meta_collect_response_instructions")
assert not hasattr(meta_prompts, "build_collect_response_instructions")
assert not hasattr(meta_prompts, "build_followup_message")
assert not hasattr(meta_prompts, "build_output_message")
```
