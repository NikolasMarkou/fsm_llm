# Executables

> Covers FSM-LLM v0.13.0

An executable is one thing that can be run by id and reports one result. Most are a JSON document that wraps an FSM body with a header: who it is, what it reads and writes, which names it needs, and which `{{variables}}` an operator fills in. This page covers the format, the two identities of a document, the rule that keeps a caller's input out of instruction text, the load sequence that checks a document, how an executable runs (the run context, the protocol and the FSM-body adapter), the registry that serves executables by id, the built-in agent patterns and reasoning strategies, the workflow step that runs one, the `call` state that runs one as a frame, and the limits that are known.

What runs today: a document is parsed and checked, wrapped by `FsmBodyExecutable` (the kinds `fsm`, `strategy` and `procedure`) or implemented in Python (the built-ins), served by an `ExecutableRegistry`, and run with `run` or `arun` under a `RunContext`. The result is a [`RunResult`](results_and_names.md) that carries both identities. Nothing in a document executes code or imports a module: a document names its tools, handlers and callables, and the host supplies them.

## The document

```json
{
  "schema": "fsm-llm/executable",
  "schema_version": "1.0",
  "kind": "fsm",
  "id": "support/greeter",
  "version": "1",
  "name": "Greeter",
  "description": "Greet a named person, then say goodbye",
  "contract": {"inputs": ["name"], "outputs": ["name"]},
  "variables": ["company"],
  "body": {
    "fsm": {
      "name": "Greeter",
      "description": "Greet a named person, then say goodbye",
      "initial_state": "greet",
      "persona": "A receptionist at {{company}}",
      "states": {
        "greet": {
          "id": "greet",
          "description": "Greet the person by name",
          "purpose": "Welcome the person",
          "response_instructions": "Greet the person by the name in the context",
          "transitions": [
            {
              "target_state": "goodbye",
              "description": "Always, after the greeting",
              "priority": 100
            }
          ]
        },
        "goodbye": {
          "id": "goodbye",
          "description": "Say goodbye",
          "purpose": "Wrap up",
          "response_instructions": "Say goodbye by name"
        }
      }
    }
  }
}
```

The header fields:

| Field | Meaning |
|---|---|
| `schema` | The constant `fsm-llm/executable`. It is how a tool tells an executable from a plain FSM file. |
| `schema_version` | The version of this envelope format, `N.M`, default `1.0`. It is not the FSM definition version and not the author's `version`. This release reads `1.0` up to any `1.x`; another major or a lower number is refused. |
| `kind` | `fsm`, `strategy` or `procedure`. It decides the shape of `body`. A document of kind `pattern` or `workflow` is refused by name: the built-in patterns are code, not documents (see [Built-in executables](#built-in-executables)). |
| `id` | A name of letters, digits and `_ . : / -`, at most 128 characters, starting with a letter or digit. |
| `version`, `name`, `description` | The author's own label and texts, all optional. |
| `contract` | `inputs` (keys the caller supplies), `outputs` (keys the run returns), `handler_only_keys`, and `input_templates` (see below). |
| `requires` | The names the body uses: `tools` (a list, or the bare claim `"none"` or `"some"`), `handlers`, `callables`, `executables`. |
| `capabilities` | A list drawn from `tools`, `hitl`, `streaming`, `strategy` and `conversational`. |
| `lifecycle` | `draft` (the default), `verified`, `trusted`, `deprecated` or `builtin`. |
| `variables` | The operator-filled `{{name}}` placeholders (lower case letters, digits and `_`, starting with a letter). |
| `metadata` | Free-form author data. |
| `body` | By kind: `{fsm}`, `{fsm, strategy}` (a strategy declares `aliases`, `results` and a `solution_key`) or `{fsm, provenance}`. |

Every model forbids a key it does not know, so a misspelt field is an error and not silently dropped.

`lifecycle`, `capabilities`, `contract` and `requires` are claims of the document. Nothing treats them as authorization: a document that says `trusted` is not trusted for it, and the checks below read the claims to compare them with the body, not to grant anything.

`body.fsm` is a plain JSON object here. It is checked by the same validator as an FSM file and loaded exactly as `API` loads a definition, so an FSM written for `API.from_file` can be wrapped without changes.

The generated JSON Schema of the format ships with the package as `fsm_llm/core/definitions/schemas/executable.schema.json` (package data). `fsm_llm.core.definitions.executable.executable_json_schema()` builds the same schema from the models, and a test holds the file equal to it. The schema describes the header; for `body.fsm` it says only "an object" because the FSM is judged by the validator, not by the schema. The bytes of the file depend on the pydantic version that generated it, which is pinned in `constraints.txt`.

## Two identities

A document has two digests, and a run can report both (the `identity` field of a [`RunResult`](results_and_names.md)):

- The **template identity** is the 64-hex sha256 of the canonical dump of the whole document: by alias, with every field still at its default left out, keys sorted. Adding a defaulted field in a later release does not move it.
- The **instance identity** is the `definition_digest` of the FSM after the variables are filled: the full digest of which the engine's `fsm_id` keeps the first characters.

```python
from fsm_llm.core.definitions.executable import identity_of, instantiate_envelope, parse_executable

envelope = parse_executable(document)                  # V1, see below
definition, record = instantiate_envelope(envelope, variables={"company": "Acme"})
identity = identity_of(envelope, definition)           # template_identity, instance_identity
# record == {"options": {}, "variables": {"company": "Acme"}}
```

`instantiate_envelope(envelope, *, options=None, variables=None)` replaces `{{name}}` in the string values of `body.fsm` for the declared variables. It never touches a key, never rescans a replacement, and leaves an unknown `{{name}}` as text. It raises `InstantiationError` for a declared variable with no value, a value for a name that is not declared, a value that is not text, a non-empty `options` (no kind has an option schema yet), or a body that is not a valid FSM definition. The returned `record` is what you log. `rebuild_instance(envelope, record, expected_instance_identity)` re-instantiates from a record and raises `InstanceIdentityMismatchError` unless the digest equals the one that was logged, so a replay can prove it ran the same body.

`identity_of(template, instance=None)` returns an `Identity` with `instance_identity` left `None` when no instance is given. The instance may be an `FSMDefinition` or a dict: a dict that loads as an FSM definition gives the same instance identity as the definition (one instance, one identity), and only a dict that is not an FSM is hashed as written. Parse and instantiation functions raise `ExecutableDocumentError` and `InstantiationError`, both `ValueError`s and `FSMError`s.

## A caller's input is data, never instruction text

The rule behind the `variables` and `contract.input_templates` split: text a caller supplies at run time may become the value of an input, but it never becomes part of an instruction, a persona or any other text the model is told to obey.

- `variables` are filled by the operator when the template is instantiated. They are part of the instance, so they are part of its identity.
- `contract.inputs` are filled by the caller per run, as data. `contract.input_templates` maps an input name to a text with `{{placeholders}}` that is rendered from the caller's arguments into that input's value, and only there.
- A variable may not be named like an input or like a placeholder of an input template. An `input_templates` key must be a declared input. Both are refused when the document is parsed.
- A body that uses `{{x}}` in its text, where `x` is a caller input, is an error at load (`[inputs]`).

A caller's task therefore never enters either identity function: no function that computes an identity takes a caller input.

## The load sequence

`load_executable` runs the checks in order and reports every failure it can find in one result, each message tagged by its step:

```python
from fsm_llm.core.verify.executable import load_executable

envelope, result = load_executable(
    document,
    variables={"company": "Acme"},   # None: check the template as written
    resolver=None,                   # Callable[[str, str], bool], see V3
    origin="operator",               # or "model", see V7
    allow=None,
)
result.is_valid, result.errors, result.warnings, result.info
```

It returns `(envelope or None, FSMValidationResult)`. The envelope is returned only when the result is valid. A document that is not an object, is nested too deep (more than 200 levels) or is refused by the models is an invalid result, never an exception.

| Step | Tag | What it checks |
|---|---|---|
| V1 | `[schema]` | The document against the models. A failure here ends the call: nothing else can be read. |
| V4 (values) | `[variables]` | With `variables` given: every declared variable has a text value and no value is undeclared. The FSM body is then checked with the values filled in. With `variables=None` the body is checked as written and an info line says the variables are not filled. |
| V2 | `[body]` | `FSMValidator` over the body. Its errors, warnings and info are copied in. |
| V4 (text) | `[inputs]`, `[variables]` | A `{{caller input}}` in body text is an error. A `{{name}}` that is not declared, or a declared one that sits in a key (keys are left alone), is a warning. |
| V5 | `[contract]` | The data-flow and liveness analyses against the declared inputs and outputs. These are warnings only, so a document is never refused for a finding the analysis can get wrong. |
| V3 | `[names]` | Completion tool names in the raw states must be listed in `requires.tools` (when it is a list). With a `resolver`, every name of `requires.tools`, `handlers`, `callables` and `executables` must resolve. With none, one info line says names were not resolved. |
| V7 | `[model]` | Only when `origin="model"`. See below. |

The numbering follows the roadmap: V6 (a strict mode) and the V4 variants for the pattern kind are not built.

The `resolver` is a function called with `(category, name)`, the category being `tools`, `handlers`, `callables` or `executables`. The loader reads no registry itself; build the function from your [name registries](results_and_names.md#name-registries): `lambda category, name: name in namespaces[category]`. The `fsm-llm-validate` command passes no resolver, so it checks that the names are declared and not that they resolve; `ExecutableRegistry.add_document` takes a `resolver` of its own.

### Text written by a model (`origin="model"`)

`origin` says who wrote the document. `"operator"` is a file a person wrote and V7 never applies. `"model"` is text a model produced, and V7 then refuses it when:

- a `requires` name of `tools`, `callables` or `executables` is not on the `allow` list (`allow` maps a category to the names a model may require; `None` is an empty list, so every name is refused);
- a tool that a completion state of the body declares is not on the `tools` list. The body is read as written and, when variables were filled, as filled, and the check runs whatever `requires.tools` says: the claim `"some"` cannot be checked and is refused, the claim `"none"` beside declared tools is a disagreement and is refused, and a list that omits a declared tool is a `[names]` error as well;
- the body sets a `persona`, a `model` or `prompt_config` on a classification entry, a validation rule `pattern`, or has a `synthesize` state (a message names the key, never its value);
- `requires.handlers` names anything: a handler binding comes from a template, never from a model;
- `lifecycle` is anything but `draft`.

A body key that would bind code (`handlers`, `variants`, `generator`) is refused by V1 for every origin. Names, keys and state ids that came from a document reach a message only after they are capped and stripped of control characters, because a repair loop may show the message to a model. V7 has no consumer in the library yet. It is not the synthesis gate G5: V7 covers the lines above (the `requires` names, the declared tools, those four body settings, handlers and lifecycle) and nothing else of G5, so text length, key names, unknown keys, priorities and the abort edge are still checked only by G5, and a synthesis or meta-builder output needs both.

### Limits of the load sequence

- `newer_envelope_fields(envelope)` lists fields newer than the declared `schema_version`, but `load_executable` does not call it. Only version `1.0` exists, so there is nothing newer to report yet.
- The contract check (V5) warns and never refuses.
- Claims are not authorization, as above.

## `fsm-llm-validate` and executables

```bash
fsm-llm-validate --fsm greeter.exe.json
```

The command reads the file and routes on its header: a file whose `schema` is `fsm-llm/executable` is validated as an executable, anything else as an FSM definition, whatever the extension. The report is the same JSON as for an FSM (`fsm_name` is the document `id`, then `is_valid`, `errors`, `warnings`, `info`), with the step tags in each line. It checks the template as written (no variables given) and resolves no names. The exit code is 0 for a valid document and 1 otherwise.

`--upgrade` rewrites an older FSM definition file. An executable is never rewritten: with `--upgrade` the command prints the report, logs one warning and exits 1, and no file changes.

## Running an executable

### The run context

`RunContext` (`fsm_llm.engine.context`) is the frozen value a runner takes for one run. It is imported by name; `import fsm_llm` does not load it, and no `API` constructor reads it.

| Field | Meaning |
|---|---|
| `llm_interface` | The model interface of the run. |
| `llm_factory` | A function with no arguments that returns an interface, called once per run. It is refused together with `llm_interface`. With neither, the run resolves a model the way `API` does, from `LLM_MODEL`. |
| `event_log` | What `API(event_log=)` takes: an `EventLog` or a directory path. |
| `max_steps`, `max_seconds` | The limits of one run, checked by the same rule as `API`: `max_steps` is an `int` of at least 1, `max_seconds` a number above 0, and neither is a `bool`. `None` means no limit from the context. |
| `cancel` | A `CancelToken`; a fresh one by default. |
| `callables` | A `NameRegistry` of the host objects a code-backed executable needs (see [Built-in executables](#built-in-executables)). |
| `parent` | The reference (a conversation or instance id) of the run that asked, or `None`. |

A bad value raises `ValueError` (`TypeError` for a `cancel` or `callables` of the wrong type) when the context is built. `resolve_llm()` returns the interface, one fresh `llm_factory()` result, or `None`.

`context.derive(parent=..., max_steps=None, max_seconds=None, cancel=None)` returns a child context that shares everything else (the cancel token too, unless a `CancelToken` is given as `cancel`: then the child has its own, and cancelling it leaves the parent's alone), names `parent`, and narrows the limits it is given: a value above the parent's own raises `ValueError`, and `None` keeps the parent's limit, it never removes one. The limits are per run, not a shared ledger: a child's steps are not subtracted from the parent's.

`CancelToken` is one thread-safe, set-once request: `cancel()`, `cancelled`, and `raise_if_cancelled()`, which raises `RunCancelledError` (an `FSMError`). A run checks it between steps. It never interrupts a model call that is already in flight, and it is not an `asyncio` cancel: a cancelled task still raises `CancelledError`, and the engine rolls its turn back whole.

### The protocol

An `Executable` has `id`, `kind`, `version`, `contract`, `requires`, `capabilities`, `lifecycle` and `template_identity`, and three methods:

- `instantiate(options=None, variables=None)` returns an `Instance`: the body with its variables filled, with `record` (the instantiation record) and `instance_identity` (`None` when the body is code and has no static instance).
- `run(inputs, *, context=None)` and `arun(inputs, *, context=None)` run the executable and return a `RunResult`. `run` is the thin sync name over `arun` (`run_sync`), so a sync caller needs no event loop. On an executable, `run` is the run of the default instance, with no variables: a document that declares variables must be instantiated with values first, or `run` raises `InstantiationError`.

An instance is built per run and no registry keeps it. Both `run` and `arun` raise `ValueError` for an input key the contract does not declare, or one with an internal prefix, before any model call.

### An FSM-body executable

`FsmBodyExecutable(envelope, *, setup=None, pins=None)` runs a parsed envelope of kind `fsm`, `strategy` or `procedure`. `setup(api)` is an optional function called once per run on the fresh `API`, before the conversation starts: the strategies that need handlers register them there.

```python
from fsm_llm.core.definitions.executable import parse_executable
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import FsmBodyExecutable

executable = FsmBodyExecutable(parse_executable(document))
instance = executable.instantiate(variables={"company": "Acme"})
result = instance.run(
    {"name": "Ada"},
    context=RunContext(llm_interface=llm, max_steps=20, max_seconds=60),
)
result.status, result.outputs, result.answer
result.identity        # both digests of this run
result.handle          # the conversation id the run used
```

A run does this, in order:

1. Checks the input keys (above). A declared input that is missing is not an error but a result: `failed` with the reason `inputs_missing`.
2. Builds a fresh `API` for the instance with the interface and the event log of the context, calls `setup(api)` if there is one, and starts a conversation with the inputs as its initial context.
3. Drives the body with the message-free step loop (`arun_until_terminal`): no user message is sent. The step budget is `max_steps` of the context, or 100 (`DEFAULT_EXECUTABLE_MAX_STEPS`) when it sets none, and `max_seconds` bounds the time. The cancel token is checked before every step.
4. Ends the conversation and closes the `API` in a `finally`, once, whatever the outcome.

The outcome maps to a result:

| What happened | `status` | `reason` |
|---|---|---|
| The body reached a terminal state and every declared output is there | `ok` | none |
| The body reached a terminal state and a declared output is missing | `failed` | `return_keys_missing` |
| A declared input was not given | `failed` | `inputs_missing` |
| The step or time budget was spent | `aborted` | `budget_spent` |
| The cancel token was set | `aborted` | `cancelled` |
| The provider failed (`LLMResponseError`) | `failed` | `provider_error` |

Any other exception propagates, and an `asyncio.CancelledError` is never caught. `outputs` holds the declared output keys (a strategy with none declared returns its mapped result keys), `answer` the last reply the run spoke (the opening reply when no step spoke), and `detail` the kind, the number of steps and the final state. A strategy's `answer` is its solution when there is one.

## The registry

`ExecutableRegistry` (`fsm_llm.engine.executables`) serves executables by id.

```python
from fsm_llm.engine.executables import ExecutableRegistry

registry = ExecutableRegistry.from_paths(["executables/"])   # files, or folders of *.json
registry.add_document(document)                              # one more, checked by load_executable
executable = registry.get("support/greeter")                 # the greatest version
registry.get("support/greeter", version="1")
registry.list(kind="fsm", launchable=True)         # also capability="hitl" and the like
```

- **Adding.** `add_document(data, *, variables=None, resolver=None)` runs the [load sequence](#the-load-sequence), raises `ExecutableError` with every error of an invalid document, and builds the executable through the adapter of its kind. `register(executable)` adds one that is already built, such as a code-backed executable. Two executables with the same `(id, version)` are refused with `ExecutableCollisionError`.
- **Path sources.** `from_paths(paths)` and `reload()` read files, and the `*.json` files of folders in sorted order. Each file is read as UTF-8 (a byte order mark is accepted), the `.json` suffix is matched without regard to case, at most `MAX_DOCUMENT_BYTES` (2 MiB), and a folder member is confined to its folder (`security.resolve_within`); a symbolic link is reported and skipped. `reload()` validates every source first and then swaps one immutable snapshot, so a reader sees a wholly old or a wholly new view, never half of each. A file that fails keeps the entries it had and is reported, and so does a folder that cannot be listed (all its entries stay, reported under the folder's resolved path); a file that is gone drops out. For a duplicate `(id, version)` the first source wins, and a numeric version label equal to one already served as an integer dotted tuple (`01` beside `1`) is refused the same way, so `get(id)` never depends on the order of registration. `ReloadReport` names the ids served, the failures per source and the refusals.
- **Reading.** `get(id, version=None)` returns the greatest version, ordered as dotted integers (`"2"` above `"1.5"`); a version label that is not dotted integers is allowed only as the sole version of its id. `get_by_template(identity)` finds an executable by its template identity. A miss raises `ExecutableNotFoundError`, or `ExecutableEvictedError` for an identity that is neither current nor retained. `list(kind=None, capability=None, launchable=None)` is sorted by id and version; an executable is `launchable` when its `requires.callables` is empty, which is what a server or a launcher may start without a host that supplies objects.

### Retention

The registry keeps templates and never keeps an instance.

- The current snapshot holds the executables of the sources. Every `get`, `get_by_template`, `register` and reload also touches a bounded least-recently-used set of templates (`MAX_RETAINED_TEMPLATES`, 256, or `max_retained_templates=` on the constructor), so a template that a reload replaced stays reachable by its identity while it is recent.
- An executable built through the registry pins its template for the length of every run (`pin` and `unpin` are counted and public). A pinned template is never evicted, so a run that started before a reload finishes on the template it started with, while new `get` calls see the new one.
- `rebuild(template_identity, record, expected_instance_identity)` returns the instance of a logged run, rebuilt from its retained template and its instantiation record. It raises `InstanceIdentityMismatchError` unless the rebuilt digest equals the logged one, and `ExecutableEvictedError`, naming the identity, when the template is gone: the registry never guesses.
- A server that runs a million distinct tasks therefore holds one template and no instances. A caller's task is an input, not part of the template.

Pins and the retained set are in memory. A restart forgets them, and nothing on disk keeps a template.

### Aliases

An alias is one name served by several versions, with weights:

```python
registry.set_alias("greeter", {"1": 90, "2": 10}, target="support/greeter")
served = registry.resolve("greeter", experiment="tone-test", conversation_id=conversation_id)
served.executable, served.template_identity, served.route
```

A route key is a version label of `target` (the alias name when `target` is omitted) or a 64-hex template identity. Weights are finite numbers of at least 0 that sum above 0. With several live routes both `experiment` and `conversation_id` are required, and the route is a pure function of them: the first 64 bits of `sha256(f"{experiment}:{conversation_id}")` placed on the cumulative weights of the routes sorted by key. The same pair always gets the same route, and a route of weight 0 is never picked. `resolve(..., pinned_template_identity=...)` returns that template as it is and never rolls again, which is how a resumed conversation keeps what it was served. A name that is not an alias resolves to the plain id. Aliases are held in memory and set by code; no file defines one, and `resolve` writes no event.

### The two namespaces

Two [name registries](results_and_names.md#name-registries) hold the extension points, both with an entry-point group:

- `EXECUTABLE_KINDS` (`executable_kinds`, group `fsm_llm.executable_kinds`) maps a document `kind` to the adapter factory that turns a parsed envelope into an executable. `fsm`, `strategy` and `procedure` are registered, each to `FsmBodyExecutable`. `adapter_for(kind)` raises `ExecutableKindError` for a kind with no adapter.
- `EXECUTABLE_SOURCES` (`executable_sources`, group `fsm_llm.executable_sources`) maps a source name to a function with no arguments that returns executables or envelope documents. `registry.load_sources()` calls each source that this registry has not called yet, once, and adds what it returns; a source or an item that fails is reported in the `ReloadReport` and warned, never raised. This is how `fsm_llm.agents` and `fsm_llm.reasoning` declare their built-ins without the engine importing them.

The package `pyproject.toml` declares no entry point for either group yet, so a source is registered by an explicit `EXECUTABLE_SOURCES.register(name, factory)` call, and `load_sources()` finds only the packages the process has imported. An entry point installed after the process started is not seen until it restarts. `ExecutableRegistry(kinds=..., sources=...)` takes registries of its own.

The event vocabulary of the event log declares `run/start`, `run/end` and `executable/resolve` (`fsm_llm.runtime.EventType`). Only `settle` writes the first two, for a delegated `call` (see the `call` targets section below); nothing writes `executable/resolve`, a run through `run` adds no event of its own, and the registry logs nothing. Nothing reads them back, so a pin that a restart could rebuild from the log waits for that reader.

## Built-in executables

`fsm_llm.agents` and `fsm_llm.reasoning` declare their built-ins through the `executable_sources` namespace: each registers one lazy factory, named `agents` and `reasoning`, when the package is imported, and `ExecutableRegistry().load_sources()` calls the factories of the packages your process has imported (there is no entry-point wiring yet, so import the package first). The built-ins have lifecycle `builtin`: the agent patterns under their `create_agent` names (`react`, `debate`, and so on), the nine reasoning strategies (`reasoning.deductive`, and so on) and `reasoning.solve`.

```python
import fsm_llm.agents
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry

registry = ExecutableRegistry()
registry.load_sources()
result = registry.get("debate").run(
    {"task": "Is tea better than coffee?"}, context=RunContext(llm_interface=llm)
)  # a RunResult
```

A pattern and `reasoning.solve` are not documents. Each is a code-backed executable: a subclass of `CodeBackedExecutable` (`fsm_llm.engine.executables`) that builds its agent or engine per run and runs it. The strategies are `strategy` documents on `FsmBodyExecutable`.

- A pattern's input is `task` (and an optional `initial_context`). A pattern that needs Python objects (its tools, a chain, an evaluation function, maker and checker instructions, the members of a swarm) names them in `requires.callables` and is not `launchable`: the host supplies them in `RunContext(callables=...)`, a `NameRegistry`, and an unresolved name raises `ExecutableError` before any model call. Optional objects, such as `hitl` and `config`, are read from the same registry when present. Patterns that need none (`debate`, `self_consistency`, `adapt`, `plan_execute`, `orchestrator` and `meta_builder`, with `reasoning.solve`) are launchable.
- An executable never supplies an approval. The `hitl` object you give passes through unchanged, and a tool that needs approval without one is refused when the agent is built.
- `max_seconds` of the context becomes the agent's `timeout_seconds`, but never widens a host `config`: the smaller of the two is used (a host `config` of 30 seconds under a 600 second run budget keeps 30); `max_steps` is refused for a pattern (set `max_iterations` on the config) and both are refused by `reasoning.solve`, which has its own step budget.
- `meta_builder` is single-shot: `run(task)` returns the artifact in `outputs`, and a build that does not validate is `failed`, with the errors in `detail`.
- A strategy takes one input, `problem_statement`, declares no output, and reports both identities: a standalone run returns every non-None key of the strategy's `results` map, so a refused calculator expression returns its `calculation_error_details` instead of failing the run, and a caller with no `constraints` runs. A `call` state that targets a strategy and declares no `returns` merges the same map (every non-None mapped child value, a key the child never wrote simply absent); frame mode reads no contract, the call's own `inputs` are what the child receives. `simple_calculator` and `hybrid` run with the handler registration of the reasoning package as their `setup`. A pattern and `reasoning.solve` report the template identity only: their instance identity is `None`, because the body is code and the task sits in the FSM description until patterns become files.
- Budget and provider errors map as for an FSM body: a spent agent budget or timeout, or a spent reasoning budget, is `aborted` with `budget_spent`; a provider failure of `reasoning.solve` is `failed` with `provider_error`; any other error propagates.

To add your own, subclass `CodeBackedExecutable` and implement `execute`:

```python
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions.executable import Contract
from fsm_llm.core.results import RunResult
from fsm_llm.engine.executables import CodeBackedExecutable


class Shout(CodeBackedExecutable):
    def __init__(self) -> None:
        super().__init__(
            id="demo/shout", kind="pattern", impl="myapp.Shout",
            contract=Contract(inputs=["text"], outputs=["shouted"]),
        )

    async def execute(self, inputs, objects, context, identity):
        return RunResult(
            status=Status.OK, outputs={"shouted": inputs["text"].upper()}, identity=identity
        )


registry.register(Shout())
```

The base checks the inputs, resolves `requires.callables` from the context into `objects` and the cancel token, then awaits `execute`. Its `template_identity` is the digest of `{kind, id, impl, the fsm_llm version, requires}`: `impl` names the code (a claim about the code revision, never about behaviour), and a caller's task is never part of it. A code-backed executable takes no variable or option, and its lifecycle is `builtin`.

## Executables in a workflow

`fsm_llm.workflows` runs an executable as a step and reads a workflow back from a document.

```python
from fsm_llm.workflows import auto_step, create_workflow, executable_step

workflow = create_workflow("greet", "Greet a customer")
workflow.with_initial_step(
    executable_step(
        "run_greeter", "Run the greeter",
        target="support/greeter", registry=registry,
        inputs={"name": "customer"},          # {executable input: workflow key}
        returns={"greeted": "name"},          # {workflow key: executable output}
        success_step_id="done", error_step_id="failed",
    )
)
workflow.with_step(auto_step("done", "Done", next_step_id=""))
workflow.with_step(auto_step("failed", "Failed", next_step_id=""))
```

`ExecutableStep` awaits `registry.get(target).arun(inputs, context=RunContext(...))`. A workflow key the context lacks is left out of the inputs, and the executable judges its own contract. `returns` copies only outputs that are not `None`, and a `returns` entry naming a workflow key with an internal prefix is refused. A result whose status is not `ok` (`failed`, `aborted`) is a failure that routes to `error_step_id` and never to the success route; with no `error_step_id` the instance fails. `max_steps` and `max_seconds` are the budgets of the one run, and the step's own `timeout_seconds` cancels the run in flight. A budget the executable refuses (`max_steps` for a pattern, either for `reasoning.solve`) is a construction error when the registry already serves the target, and a failure at run time otherwise. The step writes `executable_status`, `executable_identity` (both digests), `executable_answer` and the `executable_<step_id>_...` forms. It owns no conversation: the executable ends its own.

An executable that needs host objects (`requires.callables` is not empty) is refused unless the step names a `callables` registry (`callables=`, a `NameRegistry` of namespace `callables`), which becomes `RunContext.callables`. The step has no interface field: an FSM body takes its model from `LLM_MODEL`.

`workflow.serialize(callables=objects)` writes the registry and any function, interface or agent as its registered name, and `load_workflow(document, objects=objects)` reads it back, so a workflow with an executable step round-trips:

```python
from fsm_llm.core.names import NameRegistry
from fsm_llm.workflows import load_workflow

objects = NameRegistry("callables")
objects.register("executables", registry)
document = workflow.serialize(callables=objects)
assert load_workflow(document, objects=objects) == workflow
```

The loader resolves a step `type` only through the built-in step classes and the classes you pass, matches names only through `objects`, and reports every failure of a document in one `WorkflowValidationError`. See the [API Reference](api_reference.md#workflowengine-fsm_llmworkflows).

The model-calling steps are awaited on the loop: a `ConversationStep`, an `AgentStep` whose agent has its own `async def arun` and an `LLMStep` over a real interface run with no worker thread, and their time limit cancels the call in flight. `BaseAgent.arun` has a default that runs the sync `run` on a worker through `aoffload`; `AgentStep` does not count that default as native.

## Executables as `call` targets

A state of the `call` kind (see [FSM Design](fsm_design.md) and the Call state section of the [API Reference](api_reference.md)) names an executable by id and the engine runs its FSM body as a child frame of the host conversation, not through `run`. The registry is the resolver: `API(targets=registry)` (keyword-only, internal like `executors=`) asks `registry.resolve_target(id)`, which answers a `ResolvedTarget(definition, template_identity, instance_identity, mode="frame", returns=...)` or `None`.

- **What resolves.** The greatest version of a plain id of kind `fsm`, `strategy` or `procedure` that instantiates. `None` (the call ends `failed/unknown_target`) for an unknown id, a code-backed kind (`pattern`, `workflow`), a body whose variable is unfilled (one warning naming the error type) and an id that only names an importable module (nothing is ever imported by id). Aliases and experiments are not routed by `resolve_target`.
- **What the child sees.** Only the `inputs` the call state declares, copied; `contract.inputs` is not read in frame mode. The child never receives the parent's context, an internal key or an approval grant. The parent receives only the declared `returns`, and for a `strategy` that declares none, every key of the strategy's own `results` map that the child wrote (`ResolvedTarget.returns`).
- **What the record carries.** The final `call_result` holds the template identity of the executable and the instance identity of what was pushed (the identities `run` would report for the same inputs), read from what `settle` pushed and never from the forgeable public record; after a restart that cannot prove it they are `None`.
- **Nested runs.** `FsmInstance.arun` gives its own `API` the context's registry (`RunContext.registry`), so a body with a `call` state runs under a run; a body with a call state and no registry is refused.
- **Delegated mode (code as a target), built for patterns.** A target that is code (a pattern) cannot be a frame. With `API(targets=ExecutableTargets(registry, context))` (a resolver that is also a `RunLauncher`) the call runs it off the turn on a daemon thread, and a registry alone still ends such a call `failed/unknown_target`. Pass 1 reserves a minted handle and writes `call_result {status: running, mode: delegated, handle, deadline, ...}`; `settle` starts the run after the turn commits, from the parent's own state and only from that reservation, and writes `run/start` first (an fsynced checkpoint, `source: run`, the parent frame id); on every later turn the executor polls the host, and a finished run lands as ONE write (the final `call_result` and the declared `returns`, each reading a declared output of the run or the pseudo output `answer`, capped at `MAX_DELEGATED_RETURN_CHARS`). `aadvance` and the step streams (`advance_stream`, `aadvance_stream`, so also `run_until_terminal_stream`) make one bounded wait (`DELEGATED_WAIT_SECONDS`, 0.5) after their step so a run loop does not spin. The call's `max_seconds` defaults to 600 and is capped at 3600 (`DEFAULT_DELEGATED_MAX_SECONDS`, `MAX_DELEGATED_MAX_SECONDS`); past it the call ends `aborted/budget_spent` and the run is only detached. A handle the host never saw for that conversation (a restart, a restored snapshot, a fork, a forged `call_result`) ends `failed/run_lost` and nothing is re-run. The child receives only the declared `inputs`, never the parent's context or an approval grant. `run/end` is written by `settle` on the engine thread when the committed record is final, never by a worker. Only a built-in pattern resolves, and only when the host context supplies its `requires.callables` (a `react` call needs a `tools` callable in `ExecutableTargets(registry, context).context`), it is not `conversational`, the call's `inputs` are inputs of its contract (`task`, and `initial_context` for most) and the call sets no `max_steps` for a pattern (a pattern refuses it): anything else ends `failed/unknown_target`. A call's `max_steps` reaches an executable that accepts a step budget (`reasoning.solve` receives it, narrowed against the host context's own `max_steps`). The child runs on the host context's interface, callables and event log, never on the parent's context, so two delegated runs share that interface on two worker threads (a sync-only interface must tolerate it; the usage meter of a `LiteLLMInterface` is locked and loses no call). A target of kind `workflow` is not built and ends `failed/unknown_target`: delegated workflows wait for workflow persistence and `Executable.resume` (roadmap E5.2 and E5.3).
- **A `procedure` target is not gated.** The call kind trusts the ids its author or its closed `choose` list names; it checks no provenance and no lifecycle but `deprecated`, which resolves to nothing and ends the call `failed/unknown_target` (the `synthesize` kind judges both, through its executor). Do not list a model-written procedure where an untrusted value can reach the list.

## Known limits

Each limit has the condition under which it applies.

- **No stall detection.** A body that needs a user message to leave a state (a transition gated on a value only a user supplies) runs message-free steps until the step budget is spent, and ends `aborted` with `budget_spent`. The run is bounded, never stuck. Give such a body its inputs as contract inputs, or do not run it as an executable.
- **Templates are bounded and pins are in memory.** The registry retains at most `MAX_RETAINED_TEMPLATES` templates beside the current ones. A template that was replaced, evicted and is not pinned cannot be found by its identity (`ExecutableEvictedError`), and a restart forgets every pin. A run that started and never ended is not pinned from the log: a delegated `call` writes `run/start` and `run/end`, but no reader turns them into a pin yet.
- **Cancel hazard of the code-backed built-ins.** A pattern and `reasoning.solve` run their synchronous body on a worker thread (`aoffload`). A cancelled `await` returns at once, but the worker keeps running to its end: its tools and handlers are not stopped, and its late result is dropped. The cancel token is checked only before the run starts. The same holds for an `AgentStep` whose agent has no `arun` of its own, and for the default `BaseAgent.arun`: it is one worker hop, not an async body.
- **A code-backed pattern has no instance identity.** Its `RunResult.identity` carries the template identity and `instance_identity=None`, because the task sits in the FSM description and the per-run tools shape the instructions. A caller's task never enters either identity.
- **A delegated run is not interruptible and not resumable.** A deadline, a cancel, `end_conversation` and `close()` only set the run's token, which a pattern checks between steps: a detached pattern may keep running and calling tools on its daemon thread, and nothing joins it. After a restart, a snapshot restore or a fork the run is lost (`failed/run_lost`) and nothing is re-run, because the host scopes a handle to the conversation that reserved it: even a fork on the same `API`, taken while the run was in flight, adopts nothing; a fork taken after the final record is a normal fork. The mark that owes `run/end` is in memory, so after a crash a log with a `run/start` and no `run/end` stays that way, even after the resumed process wrote the final `failed/run_lost` record, and so does a log whose process died between the turn that committed the final record and the append of `run/end`; a reader takes a `run/start` with no `run/end` as lost. A host context that sets `max_steps` makes every pattern fail (a pattern refuses it).
- **A detached pattern delays the exit of the process.** `PatternExecutable` runs its agent on an executor thread of the run's own event loop, and the interpreter joins executor threads at exit, so a process that exits while a pattern is still running its agent waits for what is left of that run, bounded by the agent's own timeout (the call's `max_seconds`, 600 by default, 3600 at most). A run that only awaits is killed with its daemon thread. The parent's log stays intact either way: the run thread writes no run event into it (a pattern handed the host's event log writes the files of its own agent conversations).
- **A popped frame stops its runs.** `pop_fsm` (a hand pop, or the engine's pop of a child whose budget is spent) tells the launcher to stop the runs of the popped frame and writes `run/end` (`aborted`, `cancelled`) right after `stack/pop`; a detached pattern may keep running as above. `end_conversation` drops the runs of the ended conversation without a `run/end` (its log writer is released), so a `run/start` with no `run/end` reads as lost.
- **A delegated parent spends a step per poll.** Every turn of a parent in a delegated call state polls once and is a step, and `aadvance` and the step streams wait at most 0.5 seconds for the run (one helper, `wait_for_run`, one semantics), so the `max_steps` of a run loop over it must cover the seconds of the run divided by 0.5. Leave the call state silent (no `response_instructions`) or every poll turn calls the model. `aconverse` and the message streams never wait.
- **Budgets are per run.** `RunContext.derive` narrows a limit but does not share a ledger: the steps a child spends are not taken from its parent's budget.
- **No entry-point wiring.** `pyproject.toml` declares no entry point for `executable_kinds` or `executable_sources`, so a source is found only when its package was imported and registered it.
- **No launch by id from the monitor.** The monitor still starts agents from its own table of classes and workflows from its own presets; it does not list or launch executables, and a code-backed pattern with `requires.callables` could not be launched there.
- **No `resume`, no `target` on `AgentStep` or `ConversationStep`.** An `ExecutableStep` is the only workflow step that runs an executable by id.
- **The greeting is outside `max_seconds`.** An FSM-body run starts its conversation (the greeting, the first model call when the initial state speaks) before the message-free step loop, and `max_seconds` bounds only that loop, so a slow greeting adds to it. Use the step's `timeout_seconds` (which cancels the whole run) when the total matters.
- **`ExecutableStep` does not set `RunContext.parent`.** The run is one the step starts; no parent run reference is passed, so a derived child budget or a log link to the workflow instance is not built.
- **A code-backed pattern or built-in is checked at construction only when the registry serves it.** A target a registry has not loaded yet is judged when the step runs.
- **Importing `fsm_llm.agents` or `fsm_llm.reasoning` loads the `fsm_llm.engine.executables` package** (with the run context and the document loader), because each registers its source in the namespace that lives there; the executables of the two packages load only when a registry loads the sources.
- **Claims are not authorization.** `lifecycle`, `capabilities`, `contract` and `requires` describe a document; the registry serves a `trusted` or `builtin` one exactly as it serves a `draft`.

## Next steps

- [Runs, results and names](results_and_names.md) for the `RunResult` that carries both identities and for `NameRegistry`.
- [FSM Design Guide](fsm_design.md) for the FSM inside `body.fsm`.
- [Runtime FSM synthesis](synthesis.md) for the library that stores model-written FSMs by digest.
- [Command line](cli.md) and the [API Reference](api_reference.md).
