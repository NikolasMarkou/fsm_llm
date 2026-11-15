# Agents renames and behaviour

Covers FSM-LLM 0.12.0 to 0.13.0. Three renames in `fsm_llm.agents` (the per-tool time limit, the `conv_id` parameters and the Python-tool API) and two changes of behaviour in tools and constructor limits. A rename has no alias. A behaviour change is the harder kind to find, because the old call still runs: read the last two sections even if you use no renamed name.

## timeout_s is timeout_seconds

The per-tool time limit carries its unit: `timeout_s` is renamed `timeout_seconds` on `ToolDefinition`, `PythonToolDefinition`, `@tool(timeout_seconds=)`, `ToolRegistry.register_function(..., timeout_seconds=)` and the two helpers `check_timeout` and `call_with_timeout`, in the `TOOL_BAD_TIMEOUT` error text and in a serialized `ToolDefinition` dump. `Defaults.MAX_TOOL_TIMEOUT_S` and `AgentConfig.timeout_seconds` are unchanged, and `get_json_schemas()` (so `tool_schemas_sha256`) does not carry the field.

`ToolDefinition` now sets `extra="forbid"`. A call that still passes `timeout_s=` raises a `ValidationError`; pydantic used to drop the keyword, so the tool ran with no time limit at all. A tool file loaded by `PythonToolLoader` that still says `@tool(timeout_s=...)` fails to load and is reported in `PythonToolLoadReport.errors`.

**Before**

```python
@tool(timeout_s=5.0)
def forecast(city: str) -> str:
    """Return the forecast for a city."""
    return f"Sunny in {city}"

registry.register_function(forecast_fn, timeout_s=5.0)
```

**After**

```python
from pathlib import Path

from fsm_llm.agents import PythonToolLoader, ToolDefinition, ToolRegistry, tool


@tool(timeout_seconds=5.0)
def forecast(city: str) -> str:
    """Return the forecast for a city."""
    return f"Sunny in {city}"


assert forecast._tool_definition.timeout_seconds == 5.0

registry = ToolRegistry()
registry.register_function(forecast, name="forecast2", timeout_seconds=5.0)
assert registry.get("forecast2").timeout_seconds == 5.0

try:
    ToolDefinition(name="x", description="x", timeout_s=5.0)
except ValueError as error:  # pydantic ValidationError is a ValueError
    assert "timeout_s" in str(error)

Path("tools").mkdir()
Path("tools/weather.py").write_text(
    "from fsm_llm.agents import tool\n"
    "\n"
    "@tool(timeout_seconds=5.0)\n"
    "def weather(city: str) -> str:\n"
    '    """Return the weather for a city."""\n'
    '    return f"Sunny in {city}"\n'
)
report = PythonToolLoader.load_directory("tools")
assert not report.errors
assert report.tools[0].timeout_seconds == 5.0
```

## conv_id is conversation_id

The `conv_id` parameters of `ProcedureRunner` (`hook`, `run`, `converse`, `end_conversation`) and of the agent loop hooks `BaseAgent._last_turn_cut_off`, `_end_run_conversation`, `_on_loop_iteration` and `_handle_hitl_approval` (also `VerifiedReactAgent._inject_reflection`) are renamed `conversation_id`, the spelling of core. A caller that passes it by position is not affected. A keyword call such as `runner.run(conv_id=..., ...)`, or a subclass override that keeps the old parameter name, must follow. Core calls the runner hook as `before_step(step)` only, so the rename cannot break the run loop.

**Before**

```python
results = runner.run(conv_id=conversation_id, max_steps=20)


class MyAgent(ReactAgent):
    def _on_loop_iteration(self, api, conv_id, iteration):
        ...
```

**After**

```python
import inspect

from fsm_llm.agents import ReactAgent
from fsm_llm.synthesis import ProcedureRunner

assert "conversation_id" in inspect.signature(ProcedureRunner.run).parameters
assert "conv_id" not in inspect.signature(ProcedureRunner.run).parameters
hook = ReactAgent._on_loop_iteration
assert "conversation_id" in inspect.signature(hook).parameters


class MyAgent(ReactAgent):
    def _on_loop_iteration(self, api, conversation_id, iteration):
        super()._on_loop_iteration(api, conversation_id, iteration)
```

## The Python-tool API is renamed

The Python-tool API in `fsm_llm.agents` is renamed, with no alias, no shim and no deprecation period, because the old names collide with Agent Skills (`SKILL.md` folders, which are read and never executed, while this loader executes the files it loads). Behaviour is unchanged.

| Before | After |
| --- | --- |
| module `fsm_llm.agents.skills` | `fsm_llm.agents.python_tools` |
| `SkillDefinition` | `PythonToolDefinition` |
| `SkillLoader` | `PythonToolLoader` |
| `ToolRegistry.register_skill(skill)` | `ToolRegistry.register_python_tool(tool)` |
| parameter `skills` of `PythonToolLoader.to_tool_registry` and `PythonToolLoader.by_category` | `python_tools` (a call that passes it by keyword breaks) |
| the `SKILLS = [...]` list in a `.py` file that `from_directory` loads | `PYTHON_TOOLS = [...]` |

A tool file that still uses an old name (a `SKILLS = [...]` list, a `SkillDefinition` or `SkillLoader` import) loads no tools and raises nothing: `PythonToolLoader.from_directory` skips a file that fails to load, logs a warning (library logging is off by default) and now also emits one `UserWarning` per failed file, attributed to the caller. A `SKILLS` list in a file that already imports the new names is skipped without any warning, while its `@tool`-decorated tools still load. Rename the list to `PYTHON_TOOLS` and the imports to the new names, then check `PythonToolLoader.load_directory(path).errors`.

**Before**

```python
from fsm_llm.agents.skills import SkillDefinition, SkillLoader

SKILLS = [
    SkillDefinition(name="shout", description="Upper-case text", execute=str.upper)
]
registry.register_skill(SKILLS[0])
registry = SkillLoader.to_tool_registry(skills=SKILLS, registry=registry)
```

**After**

```python
from pathlib import Path

from fsm_llm.agents import PythonToolLoader, ToolRegistry

Path("my_tools").mkdir()
Path("my_tools/shout.py").write_text(
    "from fsm_llm.agents import PythonToolDefinition\n"
    "\n"
    "PYTHON_TOOLS = [\n"
    "    PythonToolDefinition(\n"
    '        name="shout", description="Upper-case text", execute=str.upper\n'
    "    )\n"
    "]\n"
)
report = PythonToolLoader.load_directory("my_tools")
assert [t.name for t in report.tools] == ["shout"] and not report.errors

registry = ToolRegistry()
registry.register_python_tool(report.tools[0])
assert "shout" in registry.tool_names

registry = PythonToolLoader.to_tool_registry(list(report.tools))
assert "shout" in registry.tool_names
```

## A lone dict parameter has an empty schema

This one changes behaviour with no error. `@tool` and `ToolRegistry.register_function` now infer a schema by one rule, `infer_schema_from_hints` (the second pre-check in `register_function` is gone). It breaks two kinds of function:

- A `@tool` function with a lone positional dict-style parameter (`dict`, `dict[str, int]`, its string form or no annotation) now gets the empty schema on both paths, which is what `register_function` always did. The model sees an empty object schema and the whole arguments object is the dict. A `@tool` function with `tags: dict[str, int]` used to show a typed `tags` property.
- A function that names a parameter and takes `**kw` now receives the extra keys the model sends (they used to be dropped, with a warning). A lone `**kw` with an explicit schema (an MCP tool) still filters by that schema.

A keyword-only dict parameter keeps its named schema and is called by keyword; its `dict[...]` annotation is advertised as `object`. `*args` and `**kwargs` are never schema properties and never required. The description rule (first docstring line for `@tool`, the whole docstring for `register_function`) is unchanged.

If a model relied on the typed `tags` property, make the dict parameter keyword-only: it then keeps a named property.

**Before**

```python
@tool
def label(tags: dict[str, int]) -> str:
    """Label things."""
    return str(sorted(tags.items()))

# the model saw {"properties": {"tags": {...}}} and sent {"tags": {"a": 1}}
```

**After**

```python
from fsm_llm.agents import ToolCall, ToolRegistry


def label(tags: dict[str, int]) -> str:
    """Label things."""
    return str(sorted(tags.items()))


def label_named(*, tags: dict[str, int]) -> str:
    """Label things."""
    return str(sorted(tags.items()))


def echo(a: int, **kw) -> str:
    """Echo the arguments."""
    return str((a, kw))


registry = ToolRegistry()
for fn in (label, label_named, echo):
    registry.register_function(fn)

assert registry.get("label").parameter_schema == {}
assert registry.get("label_named").parameter_schema["required"] == ["tags"]

whole = registry.execute(ToolCall(tool_name="label", parameters={"a": 1}))
assert whole.observation == "[('a', 1)]"
named = registry.execute(
    ToolCall(tool_name="label_named", parameters={"tags": {"a": 1}})
)
assert named.observation == "[('a', 1)]"
extra = registry.execute(ToolCall(tool_name="echo", parameters={"a": 1, "b": 2}))
assert extra.observation == "(1, {'b': 2})"
```

## Constructor limits refuse bad values

Every `max_*` and `num_*` limit of the agent constructors goes through one helper, `require_int_limit(name, value, *, minimum, error=AgentError)` (in `fsm_llm.tools`). A caller that relied on a silent fix-up now gets an error. The helper accepts an `int` (or a numpy integer, stored as a plain `int`) from the minimum to `Defaults.MAX_INT_LIMIT` (2**63-1), and refuses a `bool`, a `float` (also `1.0`, NaN and infinity), a `str`, `None`, a value below the minimum and a larger int. The message holds only a short prefix of the value. Before, `num_rounds=10**5000` raised a bare `ValueError` from a log line.

The minimums are 1 for `max_reflections`, `max_parallel`, `num_rounds`, `max_workers` (Orchestrator and SelfConsistency) and `num_samples`, and 0 for `max_replans`, `max_refinements`, `max_revisions`, `max_depth` (ADaPT), `max_handoffs` and `max_verify_retries` (0 means none of that work, so it stays valid). `DebateAgent(num_rounds=0)` and a negative value are now an `AgentError`; they were clamped to 1 silently.

The same rule, with `ValueError` kept, now covers `AgentServer(max_concurrent=, max_input_chars=)` (minimum 1), `RetryingToolRegistry(max_retries=)` (0), `CachingToolRegistry(max_entries=)` (1; 0 used to raise `StopIteration` from the first cached call) and `SemanticMemoryStore(max_entries=)` (0). A 5000-digit tool timeout is now a `ToolExecutionError`, not a `ValueError`.

**Before**

```python
agent = DebateAgent(config=config, num_rounds=0)  # clamped to 1, no error
agent = DebateAgent(config=config, num_rounds=2.0)
```

**After**

```python
from fsm_llm.agents import AgentConfig, AgentError, DebateAgent
from fsm_llm.tools import require_int_limit

config = AgentConfig(model="ollama_chat/example")
DebateAgent(config=config, num_rounds=1)
for bad in (0, -1, True, 2.0, "3", None):
    try:
        DebateAgent(config=config, num_rounds=bad)
    except AgentError:
        pass
    else:
        raise AssertionError(f"num_rounds={bad!r} was accepted")

assert require_int_limit("num_samples", 5, minimum=1) == 5
assert require_int_limit("max_replans", 0, minimum=0) == 0
```
