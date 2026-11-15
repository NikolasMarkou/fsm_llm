# Tools and approval policy moves

Covers FSM-LLM 0.12.0 to 0.13.0. The tool layer and the human-in-the-loop (HITL) approval layer left `fsm_llm.agents` and became two layer packages of their own: `fsm_llm.tools` (the registry, the `@tool` decorator, the tool models, the tool errors and the truncation helper) and `fsm_llm.policy` (the approval types and `HumanInTheLoop`). Both ship with the core install and import no agent module, so a host that builds its own tool loop no longer needs the agents package.

Nothing is lost for code that imports from `fsm_llm.agents`: every public name of these moves is still exported by `fsm_llm.agents` (`from fsm_llm.agents import ToolRegistry` is unchanged). What breaks is the old deep module path (`fsm_llm.agents.tools`, `.definitions`, `.exceptions`, `.truncation`, `.hitl`) for a name that moved, and a patch or log filter that names it. None has an alias, so the old import fails at once with an `ImportError`. The per-module symbol tables are on the [module paths](module-paths.md) page.

## ToolRegistry and the tool helpers moved to fsm_llm.tools

`ToolRegistry`, `tool`, `redact_secret_entries`, `tool_parameters_schema` and `schema_types` (present in `fsm_llm.agents.tools` in 0.12.0) are defined in `fsm_llm.tools` and no longer in `fsm_llm.agents.tools`. `register_agent`, `normalize_tool_input`, `refuse_execute_without_gated` and `tool_call_from_context` stay in `fsm_llm.agents.tools`. The helpers added to the same module after 0.12.0 (`registry_from`, `check_timeout`, `call_with_timeout`, `ToolTimedOut`, `callable_name`, `infer_schema_from_hints`) were never tagged and live in `fsm_llm.tools` only.

A patch aimed at `fsm_llm.agents.tools.<name>` for a moved name, or for a module attribute the old module only imported (`time`, `logger`, `redact_context`, and so on), must aim at the module that performs the lookup, for example `fsm_llm.tools.registry.time` or `fsm_llm.tools.timeouts.logger`.

**Before**

```python
from fsm_llm.agents.tools import ToolRegistry, tool
```

**After**

```python
from fsm_llm.agents import ToolRegistry as AgentsToolRegistry
from fsm_llm.agents.tools import register_agent
from fsm_llm.tools import ToolRegistry, tool

assert AgentsToolRegistry is ToolRegistry  # the agents re-export is unchanged
assert callable(register_agent) and callable(tool)

registry = ToolRegistry()


@tool()
def echo(text: str) -> str:
    """Return the text."""
    return text


registry.register_function(echo)
assert registry.get("echo") is not None
```

## The tool models moved to fsm_llm.tools

`ToolDefinition`, `ToolCall`, `ToolResult` and `ToolAnnotations` (present in `fsm_llm.agents.definitions` in 0.12.0) are defined in `fsm_llm.tools.models` and no longer in `fsm_llm.agents.definitions`. `READ_ONLY_ANNOTATIONS` was added after 0.12.0 and lives in `fsm_llm.tools` only. `AgentConfig`, `AgentResult` and every other model of the module stay.

**Before**

```python
from fsm_llm.agents.definitions import ToolCall, ToolDefinition
```

**After**

```python
from fsm_llm.agents import ToolCall as AgentsToolCall
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.tools import ToolCall, ToolDefinition

assert AgentsToolCall is ToolCall
call = ToolCall(tool_name="echo", parameters={"text": "hi"})
assert call.tool_name == "echo"
assert ToolDefinition.__module__ == "fsm_llm.tools.models"
assert AgentConfig.__module__.startswith("fsm_llm.agents")
```

## The tool errors moved to fsm_llm.tools and AgentError to core

`ToolExecutionError` and `ToolNotFoundError` (present in `fsm_llm.agents.exceptions` in 0.12.0) are defined in `fsm_llm.tools.errors` and no longer in `fsm_llm.agents.exceptions`. `bounded_repr` and `require_int_limit` (added after 0.12.0) live in `fsm_llm.tools` only.

`AgentError` is now defined in `fsm_llm.core.definitions.errors` (import it from `fsm_llm.core.definitions` or from `fsm_llm.agents`, both unchanged). Only its `__module__` changes. `except AgentError` still catches every agent and tool error, because the tool errors are still `AgentError` subclasses.

**Before**

```python
from fsm_llm.agents.exceptions import AgentError, ToolExecutionError
```

**After**

```python
from fsm_llm.agents import AgentError, ToolExecutionError as AgentsToolExecutionError
from fsm_llm.core.definitions import AgentError as CoreAgentError
from fsm_llm.tools import ToolExecutionError, ToolNotFoundError

assert AgentError is CoreAgentError
assert AgentsToolExecutionError is ToolExecutionError
assert issubclass(ToolExecutionError, AgentError)
assert issubclass(ToolNotFoundError, AgentError)
```

## smart_truncate moved to fsm_llm.tools

The module `fsm_llm.agents.truncation` (present in 0.12.0) moves to `fsm_llm.tools.truncation` and no longer exists. The old path does not import. `smart_truncate` is exported by the `fsm_llm.tools` package.

**Before**

```python
from fsm_llm.agents.truncation import smart_truncate
```

**After**

```python
import importlib.util

from fsm_llm.tools import smart_truncate

assert importlib.util.find_spec("fsm_llm.agents.truncation") is None
assert len(smart_truncate("x" * 5000, max_length=100)) == 100
```

## HumanInTheLoop moved to fsm_llm.policy

The module `fsm_llm.agents.hitl` (present in 0.12.0) no longer exists. `HumanInTheLoop`, `ApprovalPolicy`, `ApprovalCallback` and `EscalationCallback` are defined in `fsm_llm.policy` (in `fsm_llm.policy.hitl` and `fsm_llm.policy.approval`). `from fsm_llm.agents import HumanInTheLoop` is unchanged. `make_hitl_checker` moves to `fsm_llm.agents.tools` (it was never in `fsm_llm.agents.__all__`).

A patch aimed at `fsm_llm.agents.hitl.<name>` must aim at `fsm_llm.policy.hitl.<name>` (the manager's `logger`, `check_timeout`, `call_with_timeout`), and a log filter on the record name `fsm_llm.agents.hitl` now matches `fsm_llm.policy.hitl`.

**Before**

```python
from fsm_llm.agents.hitl import HumanInTheLoop, make_hitl_checker
```

**After**

```python
import importlib.util

from fsm_llm.agents import HumanInTheLoop as AgentsHumanInTheLoop
from fsm_llm.agents.tools import make_hitl_checker
from fsm_llm.policy import ApprovalPolicy, HumanInTheLoop

assert importlib.util.find_spec("fsm_llm.agents.hitl") is None
assert AgentsHumanInTheLoop is HumanInTheLoop
assert HumanInTheLoop.__module__ == "fsm_llm.policy.hitl"
assert ApprovalPolicy is not None  # the approval policy type moved with it
assert callable(make_hitl_checker)
```

## ApprovalRequest moved to fsm_llm.policy

`ApprovalRequest` (present in `fsm_llm.agents.definitions` in 0.12.0) is defined in `fsm_llm.policy.approval` and no longer in `fsm_llm.agents.definitions`. `from fsm_llm.agents import ApprovalRequest` is unchanged. `AgentConfig`, `AgentResult` and every other model of the module stay.

**Before**

```python
from fsm_llm.agents.definitions import ApprovalRequest
```

**After**

```python
from fsm_llm.agents import ApprovalRequest as AgentsApprovalRequest
from fsm_llm.policy import ApprovalRequest

assert AgentsApprovalRequest is ApprovalRequest
assert ApprovalRequest.__module__ == "fsm_llm.policy.approval"
```

## ApprovalDeniedError moved to fsm_llm.policy

`ApprovalDeniedError` (present in `fsm_llm.agents.exceptions` in 0.12.0) is defined in `fsm_llm.policy.approval` and no longer in `fsm_llm.agents.exceptions`. `from fsm_llm.agents import ApprovalDeniedError` is unchanged. It is still an `AgentError` (and so an `FSMError`): `except AgentError` catches it as before.

**Before**

```python
from fsm_llm.agents.exceptions import ApprovalDeniedError
```

**After**

```python
from fsm_llm import FSMError
from fsm_llm.agents import AgentError
from fsm_llm.policy import ApprovalDeniedError

assert issubclass(ApprovalDeniedError, AgentError)
assert issubclass(ApprovalDeniedError, FSMError)
```
