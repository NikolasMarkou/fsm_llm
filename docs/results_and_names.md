# Runs, results and names

> Covers FSM-LLM v0.13.0

Four small pieces of the core that every part of the library builds on: the closed vocabularies of how a run ended, the one envelope a run reports, the one registry for names, and a generic JSON patch applier. The last section says how the model is chosen. Each is imported by name; `import fsm_llm` loads `core.names` (one small module, through the engine's state-kind table) and neither `core.codes` nor `core.results`.

## Codes and the result envelope

`fsm_llm.core.codes` holds the closed vocabularies. Each is a `str` Enum with UPPER member names, the same lower case wire strings as before, and a `from_string` that raises `ValueError` for anything else:

| Enum | Members (values) |
|---|---|
| `Status` | `ok`, `failed`, `aborted`, `running`, `abstained` |
| `Reason` | `inputs_missing`, `return_keys_missing`, `provider_error`, `library_error`, `compile_failed`, `push_failed`, `approval_required`, `approval_denied`, `reviewer_error`, `depth_refused`, `completion_unsupported`, `lifecycle_too_low`, `not_simulated`, `budget_spent`, `unexpected_error`, `needs_user_input`, `cancelled`, `unknown_target`, `run_lost` |
| `Lifecycle` | `draft`, `verified`, `trusted`, `deprecated`, `builtin` |
| `MessageSource` | `user`, `run`, `remote_client`, `simulator` |
| `Purpose` | `extract`, `classify`, `tie_break`, `reply`, `complete`, `compile`, `simulate`, `judge`, `summarize` |
| `LlmFailureCode` | `timeout`, `rate_limited`, `auth`, `context_overflow`, `content_refused`, `malformed_response`, `transport`, `http_error`, `cancelled`, `unknown` |
| `ToolErrorCode` | `unknown_tool`, `invalid_args`, `execution_failed`, `timeout`, `denied`, `sandbox_violation` |

`Reason.CANCELLED` is a run stopped from outside: its status is `aborted`. `Lifecycle.BUILTIN` marks something that ships in code. Both `Lifecycle` and `MessageSource` are claims or labels, never authorization. In a message use `.value`, not an f-string of a member, because the formatting of a `str` Enum differs between Python versions.

The last three rows are the labels of the event log (see the Event Log section of the [API reference](api_reference.md)); they are records, and no code decides on them:

- `MessageSource` is written on `turn/start` of a turn that has a user message (`user` unless the caller wrapped its `converse` call in `fsm_llm.core.codes.message_source(...)`) and on `run/start`. Nothing reads it back: no prompt carries it, no approval path selects by it, and a message-free `advance` writes none.
- `Purpose` is written on `llm/request`, `llm/response` and `provider/error`. It comes from the context manager `fsm_llm.llm.purpose.purpose_scope`, or is implied by the call type when no scope is set, and it is never a field of a request, so it changes no byte sent to a provider. `tie_break` is the classifier call that settles an AMBIGUOUS transition; the same call type through `classification_extractions` is `classify`. An embedding call is not logged and has no purpose.
- `LlmFailureCode` is written on `provider/error` as `failure_code`, mapped from the exception class by `fsm_llm.llm.failures.classify_failure` (never from message text, except the one existing marker of a garbled tool call). `http_error` is any other HTTP status of 400 or more, and the number stays in the event's `status_code`. `unknown` is the fallback of the mapper.
- `ToolErrorCode` is written on `tool/result` as `error_code` and carried by `ToolResult.error_code`. `denied` and `sandbox_violation` are members that no producer writes yet (a denial writes only `approval/outcome`, and `sandbox_violation` waits for the sandbox seam of roadmap rows 4.2 and 4.3).

`fsm_llm.core.results` defines what a run reports:

```python
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.results import Identity, RunResult

result = RunResult(status=Status.ABORTED, reason=Reason.CANCELLED, handle="conv-1")
result.success                      # False: derived from status, never stored or passed
result.identity.instance_identity   # None: the identity is always an Identity, its digests may be None
```

`RunResult(status, reason, outputs, answer, detail, identity, handle)` is frozen and refuses an unknown keyword, so `RunResult(success=True)` raises. `status` and `reason` come from the vocabularies above; `outputs` holds the result values (the declared keys, for a kind that declares them); `answer` is the text, if any; `detail` carries kind-specific facts that library code never branches on; `handle` is an id to resume or inspect the run. `Identity(template_identity, instance_identity)` carries the two digests of an [executable](executables.md#two-identities); either may be `None`, meaning not known or not static for this run.

The result types of the runners project onto the envelope:

- `AgentResult.run_result()`. `AgentResult` has `status` and `reason`, and its `success` is a read-only property (`status is Status.OK`). You give a `stop_reason` and `status` and `reason` follow from it (`StopReason.ANSWERED` is `ok`; a forced stop such as `MAX_ITERATIONS` is `aborted` with `budget_spent`); `AgentResult(success=...)` is refused. `StopReason` is a `str` Enum with UPPER members and the old wire strings, so `StopReason.ANSWERED == "answered"` holds and a dump still emits `answered`.
- `WorkflowInstance.run_result()`. A pending, running or waiting instance reads `running` (the workflow status stays in `detail["workflow_status"]`), `completed` is `ok`, `failed` is `failed` with no reason (the error text rides in `detail["error"]`), `cancelled` is `aborted` with `Reason.CANCELLED`. `handle` is the instance id and `outputs` is the public non-secret context: a copy without the top-level keys that have an internal prefix and without any entry the secret filter (`is_forbidden_context_entry`) flags, at any depth. A workflow declares no output keys, so this is not a declared-key list. A context the filter cannot walk gives empty `outputs` and `detail["outputs_withheld"] = "unfilterable"`.
- `fsm_llm.synthesis.synthesis_run_result(result)` over the `synthesis_result` dict the synthesize executor writes. The dict stays the wire contract; the procedure id becomes the instance identity.
- The `call_result` a `call` state writes is the same shape of fact in a context dict, not a `RunResult`: `{status, state, reason, target, template_identity, instance_identity}` with `status` from `Status` (`ok`, `failed`, `aborted`, `running` while the child runs) and `reason` from `Reason` (`unknown_target`, `inputs_missing`, `depth_refused`, `push_failed`, `return_keys_missing`, `budget_spent`, `provider_error`), the two identities as in the envelope, and `state` the id of the call state that wrote it. See the Call state section of the [API Reference](api_reference.md).

## Name registries

An FSM definition refers to code by name only, never by object. `fsm_llm.core.names.NameRegistry[T](namespace, *, entry_point_group=None)` is the one place a name becomes an object, inside a namespace:

```python
from fsm_llm.core.names import NameRegistry

kinds: NameRegistry[object] = NameRegistry("state_kinds")
kinds.register("conversation", executor)
kinds.resolve("conversation") is executor   # True; a miss raises NameNotFoundError
kinds.name_of(executor)                     # "conversation" (by identity), or None
kinds.register("conversation", other)       # NameCollisionError: pass override=True to replace
kinds.freeze()                              # a later register raises RegistryFrozenError
```

The rules:

- A taken name is refused with `NameCollisionError` unless `override=True`. One object under a second name is always refused, so `name_of` has one answer. A name must be a non-blank string.
- `get(name, default=None)`, `resolve(name)`, `name in registry`, `list()` (sorted names), `items()` (name and object pairs in registration order; a replaced name keeps its first position), `frozen` and `namespace`.
- With `entry_point_group` set, the group is read once, on the first miss of `get`, `resolve` or `in`, or by `load_entry_points()` or `freeze()`. Each entry `name = module:attr` registers its object without overriding: an existing name wins, and a failing or colliding entry is warned and skipped. A later miss never reloads. A package installed after the process started is therefore not seen until the process restarts. The default `None` loads nothing.
- One leaf lock guards the tables. No loader, logger or callback runs under it, and a thread that misses while another loads waits for the load.
- A registry is a plain object: a module constant when global, a local when per run. There is no registry of registries.

Today the engine's state-kind table (`state_kinds`) is a `NameRegistry` with no entry-point group; the executable kinds (`executable_kinds`), executable sources (`executable_sources`), the handler factories a pattern file binds (`handlers`) and the generators it names (`generators`) are more, each with an entry-point group (`fsm_llm.executable_kinds`, `fsm_llm.executable_sources`, `fsm_llm.handlers`, `fsm_llm.generators`) that `pyproject.toml` does not wire yet, see [Executables](executables.md#the-namespaces), and `pattern_agents` is the class table of `create_agent` (no entry-point group); a workflow and a run context take the host objects as a registry of the namespace `callables`; and each `ToolRegistry` holds a `NameRegistry` named `tools`. `ToolRegistry.register` refuses a name that is already taken, where it once let the last registration win: pass `replace=True` (also accepted by `register_function`, `register_python_tool` and `register_agent`) to replace on purpose; a replaced tool keeps its position, which is the order of the tool list the model sees. Some private name tables remain (the monitor's agent classes and workflow presets, and the meta-builder artifact schemas); each is named in `tests/test_fsm_llm/test_names.py` with the roadmap row that retires it, and a test fails on a new one.

## Patching a JSON document

`fsm_llm.core.definitions.apply_json_patch(document, ops, *, max_ops, max_chars, check_path=None)` applies a patch of the subset of RFC 6902 that a model can safely write, to a trusted document, and returns a new dict (the argument is never changed):

```python
from fsm_llm.core.definitions import apply_json_patch, JsonPatchError

new = apply_json_patch(
    {"states": {"a": {"purpose": "old"}}},
    [{"op": "test", "path": "/states/a/purpose", "value": "old"},
     {"op": "replace", "path": "/states/a/purpose", "value": "new"}],
    max_ops=8, max_chars=2000,
)
```

- The ops are `add`, `remove`, `replace` and `test`. `move`, `copy` and any extra key are refused.
- A path is an RFC 6901 pointer of at most 12 tokens. `add` needs an existing parent and an index up to the list length, or `-`. The other ops need an existing target. `test` is strict equality: `1` is not `1.0` and `True` is not `1`.
- The bounds come first: `max_ops`, `max_chars` (an estimate taken in one iterative walk before any op is read) and a nesting depth of 16 for a value. A refusal raises `JsonPatchError` with a `code` (`too-many-ops`, `too-large`, `too-deep`, `bad-value`, `bad-document`, `op-shape`, `op-not-allowed`, `bad-pointer`, `bad-index`, `out-of-scope`, `no-target` or `test-failed`) and the index `op` of the op at fault. The message holds the code and the index only, never text of the patch, and nothing is applied.
- `check_path` is how a caller narrows where ops may land: a function called once per op with the unescaped tokens of the path, returning whether it may be touched. A false answer refuses the patch with `out-of-scope`. Runtime synthesis uses it to confine a repair to the states.
- The applier does not validate the result. The caller runs its own checks on the returned document.

## Choosing the model

There is one resolver, `fsm_llm.llm.resolve_model(model=None)`: the explicit name, then the environment variable `LLM_MODEL`, and otherwise `ModelNotConfiguredError`. A blank value counts as unset. The library has no built-in model anywhere: not in the core, the agents, the reasoning engine, `fsm-llm-eval` or the monitor. A caller that injects its own `llm_interface` needs no model. How each command line takes `--model` and `--env-file`, and what a monitor launch answers, is on the [Command line](cli.md) page; a machine that wants a default keeps `LLM_MODEL` in its own `.env`, never in code.

## Next steps

- [Executables](executables.md) for the envelope that carries the two identities.
- [API Reference](api_reference.md) for the signatures.
- [Architecture](architecture.md) for where these modules sit in the layers.
