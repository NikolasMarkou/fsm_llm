# Executables

> Covers FSM-LLM v0.13.0

An executable is one thing that can be run by id and reports one result. Most are a JSON document that wraps an FSM body with a header: who it is, what it reads and writes, which names it needs, and which `{{variables}}` an operator fills in. This page covers the format, the two identities of a document, the rule that keeps a caller's input out of instruction text, the load sequence that checks a document, how an executable runs (the run context, the protocol and the FSM-body adapter), the registry that serves executables by id, the `pattern` kind (agents as files: holes, variants, options, hooks and presets), the agents that run from a pattern file or a user directory, the built-in agent patterns and reasoning strategies, the workflow step that runs one, the `call` state that runs one as a frame, and the limits that are known.

What runs today: a document is parsed and checked, wrapped by `FsmBodyExecutable` (the kinds `fsm`, `strategy` and `procedure`), by `PatternExecutable` (the kind `pattern`, once `fsm_llm.agents` is imported) or implemented in Python (the code-backed built-ins), served by an `ExecutableRegistry`, and run with `run` or `arun` under a `RunContext`. The result is a [`RunResult`](results_and_names.md) that carries both identities. Nothing in a document executes code or imports a module: a document names its tools, handlers and callables, and the host supplies them.

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
| `kind` | `fsm`, `strategy`, `procedure` or `pattern`. It decides the shape of `body`. A document of kind `workflow` is refused by name. A `pattern` is a template (see [The pattern body](#the-pattern-body)); the agent patterns that have no file are code (see [Built-in executables](#built-in-executables)). |
| `id` | A name of letters, digits and `_ . : / -`, at most 128 characters, starting with a letter or digit. |
| `version`, `name`, `description` | The author's own label and texts, all optional. |
| `contract` | `inputs` (keys the caller supplies), `outputs` (keys the run returns), `handler_only_keys`, and `input_templates` (see below). |
| `requires` | The names the body uses: `tools` (a list, or the bare claim `"none"` or `"some"`), `handlers`, `callables`, `executables`. |
| `capabilities` | A list drawn from `tools`, `hitl`, `streaming`, `strategy` and `conversational`. |
| `lifecycle` | `draft` (the default), `verified`, `trusted`, `deprecated` or `builtin`. |
| `variables` | The operator-filled `{{name}}` placeholders (lower case letters, digits and `_`, starting with a letter). |
| `metadata` | Free-form author data. |
| `body` | By kind: `{fsm}`, `{fsm, strategy}` (a strategy declares `aliases`, `results` and a `solution_key`), `{fsm, provenance}` or the pattern body below. |

Every model forbids a key it does not know, so a misspelt field is an error and not silently dropped.

`lifecycle`, `capabilities`, `contract` and `requires` are claims of the document. Nothing treats them as authorization: a document that says `trusted` is not trusted for it, and the checks below read the claims to compare them with the body, not to grant anything.

`body.fsm` is a plain JSON object here. It is checked by the same validator as an FSM file and loaded exactly as `API` loads a definition, so an FSM written for `API.from_file` can be wrapped without changes.

The generated JSON Schema of the format ships with the package as `fsm_llm/core/definitions/schemas/executable.schema.json` (package data). `fsm_llm.core.definitions.executable.executable_json_schema()` builds the same schema from the models, and a test holds the file equal to it. The schema describes the header; for `body.fsm` it says only "an object" because the FSM is judged by the validator, not by the schema. The bytes of the file depend on the pydantic version that generated it, which is pinned in `constraints.txt`.

## The pattern body

A document of kind `pattern` carries a template instead of a bare FSM. Its `body` is `{fsm | generator | extends, handlers, options_schema, options, variants, run}`, and exactly one of `fsm` (an object), `extends` (a preset: see [Presets](#presets)) and `generator` (a name; a generator body takes no variants and is written by the function the caller's `generators` lookup returns for it, from the resolved options: see "A generator" below). The pattern fields are part of envelope format `1.0`. `load_executable` loads a `pattern` document through the pattern rules of the table below (V2 to V4 and V7). A registry serves one once `fsm_llm.agents` is imported: the package registers the `pattern` kind (`ExecutableRegistry.add_document` and a directory of files then give a `PatternExecutable` that runs a `PatternAgent`), and the patterns that have a file under `fsm_llm/agents/patterns/` are built-ins served from it (the list is in [Built-in executables](#built-in-executables)).

A generator is the form of a pattern whose FSM depends on its input: `prompt_chain` has one state per step and a gate edge after a gated step, so `prompt_chain.json` declares `generator: agents.prompt_chain` and the options `steps` (a list of flat objects) and `max_iterations`. A generator is `generator(options) -> dict`: it gets the resolved options and nothing else (no task, no variant, no variable), and what it writes is checked before anything runs it (a mapping, within the depth cap, a valid FSM definition) and returned as a copy. `instantiate_body`, `instantiate_envelope`, `rebuild_instance` and `load_executable` take the lookup as `generators=` (`GENERATORS.get`; with none, the body fails closed with `GeneratorUnavailableError`); `ExecutableRegistry.add_document` passes it for a pattern, so V2 builds and checks the default instance of the generator. An error names the generator and the exception type, never an option value. A model-origin document may not name a generator (V7), and no document registers one. Known limit: a step text over 10000 characters or a chain over 256 steps is refused by the options schema.

- **Holes.** A hole is the leaf `{"$hole": "<option name>"}`. It stands for one whole JSON value of any option type (an integer, a list) and replaces the leaf in place, with the key order of its parents kept. A hole is never a key, never beside other keys, never inside another hole and never the root of `fsm`; each is refused with one message at parse. Its value is inserted as given and is never rescanned for `{{name}}` text. `{{name}}` stays text-only and takes the declared `variables`.
- **Options.** `options_schema` is `{"properties": {name: spec}}`, a closed subset: `type` is `string`, `integer`, `number`, `boolean` or `array` (of one scalar type, of flat objects, or of `json` values, in `items`), with `minimum`, `maximum`, `maxLength`, `maxItems` and `enum`; no `$ref` and no object option of its own. A list of flat objects is `items: {type: object, properties: {name: scalar spec}}`: an entry has exactly the declared keys (all required, at most 16, no nesting) and is returned in the caller's key order (the tool intents of the classification variant of `react`). `items: {type: json}` declares nothing: an entry is any nested JSON value (text keys, finite numbers, at most 24 levels and 20000 values), returned as a copy (the tool schemas of `native_fc`); the defaults of `options` may be nested the same way. A value is checked strictly (a boolean is not an integer). `options` holds the default of each option that has one: an option with no default is required. An option may not be named like a contract input or an input template placeholder, as for a variable.
- **Variants.** `variants` maps a name to a list of RFC 6902 operations (`add`, `remove`, `replace`, `test`) applied to the template. The caller of `instantiate_body` selects them by name. The agent runner selects `approval` from its HITL predicate (no option can, and an option named `approval` is refused), `without_tools` for a registry that is `None` or empty, and any other variant for a boolean option of the same name that the caller sets to `true` (`native_fc` has `without_repair`). Variants apply in the order they are declared, and a refused patch names the variant and the patch code.
- **Instantiation.** `instantiate_body(envelope, options=None, *, variants=(), variables=None)` in `fsm_llm.core.definitions.executable_instance` returns the FSM as a fresh dict (two calls give equal dicts that share nothing), so key order is observable and a pattern file can reproduce an existing builder byte for byte: variants first, then one pass that fills holes and text. `instantiate_envelope(envelope, options=, variants=, variables=)` loads that dict as an `FSMDefinition` and returns the record `{"options": {resolved options}, "variables": {...}, "variants": [selected, in declaration order]}`, which `rebuild_instance` replays.
- **Identities.** The template identity is the digest of the document and never changes with the options or variants a caller picks; the instance identity is the digest of the instantiated FSM, so it varies with the options and variants that change the body and with nothing else.
- **Bindings and the run block.** `handlers` is a list of `{handler, on: entry|exit|state|context_update, states, keys, priority, params}`; `run` is `{agent_type, answer_keys, success_keys, evidence_keys, judged, run_output_keys, context_seed, step_budget, seconds_exempt, hooks}`. Core only checks their shape: the names resolve later. `step_budget`, `run_output_keys` and `seconds_exempt` are read by no runner yet: `PatternAgent` refuses a file that sets one when the agent is built (`AgentError`), it never ignores it. `context_seed` may not name an internal-prefixed or secret-looking key, so a file cannot seed a driver key or an approval grant. A hook is `{hook, at: before_run|on_iteration|after_run, params}` and names a library entry of the `handlers` namespace; the agent runner (`fsm_llm.agents.pattern_agent.PatternAgent`) calls it at its position. `before_run` takes the task and returns the task the attempts run on (applied once per `run`, in declared order). `after_run` is called with no argument when each attempt's loop ends, in declared order, also when the loop raised, and an error of a hook propagates (the orchestrator pattern re-raises the budget or timeout error a worker left in its run's holder this way; the `rewoo` pattern rewrites the answer of a cut-off plan by replacing the run's result, which the runner keeps in the handler context for the hooks); it returns `None` to accept or a `Retry` note that runs the loop again on the task followed by the note (the first `Retry` ends the chain of that attempt, and the hook owns the bound: `agents.verify_retry` counts attempts against its `max_verify_retries` option). `on_iteration` is called at every step with the step number, a reader of the conversation data and a writer, AFTER the human-in-the-loop driver of that step, and the writer refuses an internal-prefixed key and a secret-looking entry, so no hook can write the approval grant; it is built anew at each step, so it keeps nothing between steps. A `run_stream` applies `before_run` and calls the `after_run` hooks with an empty result slot; a file that binds a hook which can ask for a retry (`RETRY_HOOKS` of the handler library: `agents.verify_retry`) streams by running `run` and yielding the answer once, so a stream never ships an unverified answer, as `VerifiedReactAgent` (with a verifier) and `AutoMemoryReactAgent` do; a hook of an operator that returns a `Retry` must be named in that set. The hooks of the react family are `agents.verify_retry`, `agents.periodic_reflection`, `agents.memory_recall` and `agents.memory_remember`; `verified_react` and `auto_memory` are presets that extend `react` and add them. A model-origin document may not bind a hook (V7).

## Presets

A preset is a `pattern` document whose body has `extends: <pattern id>` instead of a template (it replaces the SOP dicts of `fsm_llm.agents.sop`, which are gone). It adds options, option defaults and run hooks to its base and nothing else: `handlers`, `variants`, `fsm`, `generator` and the other `run` fields are refused when the document is parsed, and an option named `model` is refused in any pattern (one model resolver). `load_executable(data, bases=lookup)` resolves it, where `lookup(id)` returns the base `PatternEnvelope` or `None`, and returns the FLAT envelope: the base's body with the preset applied, so the registry, `PatternExecutable` and `PatternAgent` see an ordinary pattern. The header (`id`, `name`, `description`, `version`, `lifecycle`, `metadata`) is the preset's own, a `contract` field or `requires.tools` the preset wrote replaces the base's, and `requires.handlers`, `callables`, `capabilities`, `variables` and `run.hooks` are the base's followed by the preset's. `extends` joins `requires.executables`, so it is resolved with the other names and a model-origin preset may extend only a base on `allow["executables"]`; V7 judges the document AS WRITTEN (the base is trusted because it is on the list), so a model preset may set only options its base declares. A chain may hold at most `MAX_EXTENDS_DEPTH` (4) `extends` links; a cycle, a deeper chain, an unknown base and a base with no template (a code-backed pattern) are refused by name, and a preset loaded with no `bases` is refused. A registry gives `bases` from the pattern executables it already holds, which are flat, so a chain through a registry is one link per load. The consequence is an order: a preset file of a path source finds its base only when the base is already registered, so `ExecutableRegistry.from_paths([dir])` alone reports it in `ReloadReport.failed` ("extends ..., which is no pattern with a template that can be found"); call `load_sources()` and then `reload()`, or load the directory with `load_directory_registry`, which registers the built-ins first. The template identity of a preset is the digest of the flat document, so it changes when its base changes.

The three shipped presets, `code-review`, `summarize` and `data-extraction`, extend `react`, sit beside the pattern files in `agents/patterns/` and are served by the `agent_presets` source (the same registry call as the patterns: `load_sources()`). Each declares `temperature` and `max_iterations` as options with defaults: a `PatternAgent` that stands alone (a preset or a user file; a thin class such as `DebateAgent` owns its options) applies the options named like the numeric `AgentConfig` fields on top of the config, with an option the caller gave first, a field the host's own `config` set second and the file default last (`system_prompt=` sets only `instructions`: it never turns the other config fields into host choices). `requires.tools` as a list of names makes the agent refuse a registry that lacks one. Converting an SOP by hand: `agent_pattern` becomes `extends`, `task_template` with `{var}` becomes `contract.input_templates.task` with `{{var}}`, `config_overrides` become declared options with defaults (never `model`), `required_tools` become `requires.tools`, and `output_schema` and `metadata` go to `metadata` (nothing read them).

## Agents from pattern files

`fsm_llm.agents.pattern_agent.PatternAgent(envelope, tools=None, config=None, hitl=None, options=None, callables=None, **api_kwargs)` runs a loaded `PatternEnvelope` on the run body every agent shares. Per run it instantiates the template, builds a fresh `HandlerContext` (the tools, the agent's own `hitl`, the one approval predicate, the options, the `callables`, a scope of its own) and registers the handlers the file binds, resolved by name from the `handlers` namespace in the order the file lists them. Nothing per-run lives on the agent or in a registry. The packaged classes (`ReactAgent`, `ReflexionAgent`, `DebateAgent` and the others with a file) are thin subclasses of it bound to their file; they keep their constructors, and `create_agent(id)` returns them.

- **Options.** `options=` are checked against the file's `options_schema` when the agent is built (an undeclared name or a value the schema refuses is an `AgentError` that names the option, never its value). The agent resolves them once: the caller's values plus the file's default of every other declared option, and that one mapping is what the handler context, the `before_run`, `after_run` and `on_iteration` hooks and the FSM holes read, so a default written in a file or a preset (`max_verify_retries: 3`) is the value everywhere. The default of an option named like an `AgentConfig` field is left to the config (host field first, see below). An option named `approval`, `context_keys`, `schema_field_names` or `tool_intents` is refused (`AgentError`): the first is the agent's HITL predicate, the others are filled by the agent. A standalone `PatternAgent` (a user file or a preset, not a thin class) also applies the options named like the numeric `AgentConfig` fields (`max_iterations`, `timeout_seconds`, `temperature`, `max_tokens`, `max_history_size`, `enable_prompt_cache`, `reflect_every_n`, `auto_summarize_after`) on top of the config, as the Presets section says; an option named `instructions`, `force_final_tool`, `output_schema`, `transition_config` or `verification_fn` stays a plain option of the file and never reaches the config (only a declared text variable puts text into a prompt).
- **Tools and approval.** A file that lists `tools` in `requires.callables` refuses an empty registry or none at all (`AgentError`), and `requires.tools` as a list of names makes the agent refuse a registry that lacks one. A file with `requires.tools: none` refuses `tools=` and `hitl=` with `TypeError`. A file that runs tools (its `requires.tools` is not `none`, or it binds a tool-running handler: `agents.tool_executor`, `agents.rewoo_executor` or `agents.native_fc`, the set `TOOL_RUNNING` of the handler library) and binds no `agents.approval_gate` refuses a registry that holds a tool flagged `requires_approval`, at construction and at the start of every run; a file that binds `agents.rewoo_executor` or `agents.native_fc` refuses it even beside the gate (only `agents.tool_executor` consults the approval predicate), and those two factories refuse it themselves. A claim in the document never lifts the refusal: what the file binds decides. A model-origin preset may not write `requires.tools: none`. A file that binds the gate runs the same refusal for a flagged tool nobody can approve, and a `HumanInTheLoop` given to a file with no gate is refused at construction (it could never be consulted). The grant `_approval_granted` is made only by the agent's driver, bound to one exact call: no document, option, hook, preset or binding `keys` entry can write it (an internal-prefixed or secret-looking key is refused in `context_seed` and in a binding's `keys`, and the writer an `on_iteration` hook gets refuses an internal-prefixed key and a secret-looking entry, the same rule as the rest of the framework).
- **Run fields with no reader.** `step_budget`, `run_output_keys` and `seconds_exempt` of the `run` block are refused at construction (`AgentError`); the other `run` fields and the three hook positions are honoured.
- **A file from a registry.** `create_agent(id, tools=None, *, registry=reg, config=None, **kwargs)` builds a `PatternAgent` over a `pattern` executable of the registry (`options=` and `callables=` pass through `kwargs`; `tools` is accepted when the file's `requires.tools` is not `none`). It refuses at creation what `ReactAgent` refuses: a flagged tool and a file with no `agents.approval_gate`, a gate and no `hitl` or policy to approve, a `HumanInTheLoop` and no gate. A built-in id never consults the registry, so a registry without `react` still builds `react` and a user file named like a built-in does not replace it. An id found nowhere raises one `ValueError` listing the built-in ids and the pattern ids of the registry.
- **A directory of files.** `load_directory_registry(directory, builtins, *, prog)` in `fsm_llm.engine.executables.directory` is the one loader behind the `--executables DIR` flag of `python -m fsm_llm.reasoning` and of `fsm-llm-meta`. The built-ins are registered first, so a file with a built-in id is refused with one stderr warning and never replaces it (in a registry a memory entry precedes a file whichever order they were added, so `register` after `from_paths` displaces the file and warns once). An unreadable or invalid file is one warning and is skipped, a missing directory is `FileNotFoundError`, and the file rules are the registry's (sorted `*.json` in any case of the suffix, a byte order mark accepted, a symbolic link or a file over `MAX_DOCUMENT_BYTES` skipped). `fsm-llm-meta` accepts the ids of the pattern files of the directory as `agent_type`.
- **Operator configuration.** A document of a user directory loads as operator origin: V7 does not apply, so the file may bind any handler of the `handlers` namespace, hooks, variants, options and run fields, exactly as a file passed on the command line. A document a model wrote is never loaded that way: with `origin="model"` it cannot carry handlers, hooks, variants, a generator, an options schema, `run.agent_type`, `run.run_output_keys`, `run.context_seed` or `requires.handlers`, whatever its claims say (see V7 above).
- **The `draft` convention.** A document with no `lifecycle` is `draft`, and the directory convention is that a file you or a model wrote stays `draft` until you edit it. Nothing enforces the label for a caller: a registry serves a `draft`, a `verified` and a `trusted` file alike, and only `deprecated` stops a document being a `call` target. V7 refuses any lifecycle but `draft` in a model-written document, so a model cannot promote its own file; an operator's promotion authorizes nothing.

## Two identities

A document has two digests, and a run can report both (the `identity` field of a [`RunResult`](results_and_names.md)):

- The **template identity** is the 64-hex sha256 of the canonical dump of the whole document: by alias, with every field still at its default left out, keys sorted. Adding a defaulted field in a later release does not move it.
- The **instance identity** is the `definition_digest` of the FSM after the variables are filled: the full digest of which the engine's `fsm_id` keeps the first characters.

```python
from fsm_llm.core.definitions.executable import parse_executable
from fsm_llm.core.definitions.executable_instance import identity_of, instantiate_envelope

envelope = parse_executable(document)                  # V1, see below
definition, record = instantiate_envelope(envelope, variables={"company": "Acme"})
identity = identity_of(envelope, definition)           # template_identity, instance_identity
# record == {"options": {}, "variables": {"company": "Acme"}}
```

`instantiate_envelope(envelope, *, options=None, variables=None)` replaces `{{name}}` in the string values of `body.fsm` for the declared variables. It never touches a key, never rescans a replacement, and leaves an unknown `{{name}}` as text. It raises `InstantiationError` for a declared variable with no value, a value for a name that is not declared, a value that is not text, a non-empty `options` or `variants` for a kind other than `pattern` (only a pattern has an option schema), or a body that is not a valid FSM definition. The returned `record` is what you log. `rebuild_instance(envelope, record, expected_instance_identity)` re-instantiates from a record and raises `InstanceIdentityMismatchError` unless the digest equals the one that was logged, so a replay can prove it ran the same body.

`identity_of(template, instance=None)` returns an `Identity` with `instance_identity` left `None` when no instance is given. The instance may be an `FSMDefinition` or a dict: a dict that loads as an FSM definition gives the same instance identity as the definition (one instance, one identity), and only a dict that is not an FSM is hashed as written. Parse and instantiation functions raise `ExecutableDocumentError` and `InstantiationError`, both `ValueError`s and `FSMError`s.

## A caller's input is data, never instruction text

The rule behind the `variables` and `contract.input_templates` split: text a caller supplies at run time may become the value of an input, but it never becomes part of an instruction, a persona or any other text the model is told to obey.

- `variables` are filled by the operator when the template is instantiated. They are part of the instance, so they are part of its identity.
- `contract.inputs` are filled by the caller per run, as data. `contract.input_templates` maps an input name to a text with `{{placeholders}}` that is rendered from the caller's arguments into that input's value, and only there: `render_inputs(contract, inputs, optional=())` (`fsm_llm.core.definitions.executable_instance`) runs first in both run bodies, before the input check. A template whose input the caller did not give is rendered in one pass (a value is never rescanned) from placeholder arguments that are text, numbers or booleans, and the arguments are consumed; an input given directly is used as it is; an argument that is neither an input, an optional input nor a placeholder (an internal-prefixed name is never one) is a `ValueError`, and so is a missing or non-scalar placeholder argument. A `call` state still passes only declared inputs: a delegated `call` accepts the declared and optional inputs of the contract (`task`, `initial_context`) and nothing else, so it cannot supply the placeholder arguments of an `input_templates` entry and must pass the finished text as `task` (frame mode reads no contract).
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

The numbering follows the roadmap: V6 (a strict mode) is not built. A document of kind `pattern` takes the same sequence over instances:

| Step | Tag | What a `pattern` adds |
|---|---|---|
| V2 | `[body]` | `FSMValidator` runs on the DEFAULT instance (every option at its default, no variant) and on the instance with every variant applied once in declaration order; an error of the second that the first has is not repeated, the rest is prefixed `with every variant applied:`. A variant is never checked alone. An option with no default makes the default instance impossible, so it is one error. A generator body is instantiated on its default options when the caller passes a `generators` lookup (the registry and the shipped files do) and is then checked like a template; with no lookup there is nothing to check. |
| V3 | `[names]` | Every handler named in `requires.handlers`, in `body.handlers` and in `body.run.hooks` is resolved through the `handlers` category and the generator through `generators`. With no `resolver`, a pattern that uses any such name is refused (a name that cannot be resolved does not load). The `hitl` rule: the capability `hitl`, a binding of `agents.approval_gate` and an `await_approval` state must all be present or all absent, whatever the claims say; each broken side is one error. |
| V4 | `[variables]`, `[inputs]` | A `{{name}}` that is not a declared variable is an error (for an `fsm` document it stays a warning). A hole of a free-text option (a string with no `enum`) under `description`, `purpose`, `persona`, `instructions`, `extraction_instructions` or `response_instructions` is an error: only a declared variable may put text into instruction text. |
| V7 | `[model]` | A model-origin pattern (or preset, judged as written) may carry none of `body.handlers`, `body.run.hooks`, `body.variants`, `body.generator`, `body.options_schema`, `body.run.agent_type`, `body.run.run_output_keys` and `body.run.context_seed`; each is its own message and no claim of the document lifts any of them. |

A registry adapts a loaded pattern through the `pattern` kind of `executable_kinds`, which `fsm_llm.agents` registers when it is imported; before that import `ExecutableRegistry.add_document` refuses a pattern that binds a handler (its names do not resolve: `ExecutableError`) and any other pattern with `ExecutableKindError` (no adapter).

The `resolver` is a function called with `(category, name)`, the category being `tools`, `handlers`, `callables` or `executables` (and `generators` for a pattern). The loader reads no registry itself; build the function from your [name registries](results_and_names.md#name-registries): `lambda category, name: name in namespaces[category]`. The `fsm-llm-validate` command passes no resolver, so it checks that the names are declared and not that they resolve; `ExecutableRegistry.add_document` takes a `resolver` of its own.

### Text written by a model (`origin="model"`)

`origin` says who wrote the document. `"operator"` is a file a person wrote and V7 never applies. `"model"` is text a model produced, and V7 then refuses it when:

- a `requires` name of `tools`, `callables` or `executables` is not on the `allow` list (`allow` maps a category to the names a model may require; `None` is an empty list, so every name is refused);
- a tool that a completion state of the body declares is not on the `tools` list. The body is read as written and, when variables were filled, as filled, and the check runs whatever `requires.tools` says: the claim `"some"` cannot be checked and is refused, the claim `"none"` beside declared tools is a disagreement and is refused, and a list that omits a declared tool is a `[names]` error as well;
- the body sets a `persona`, a `model` or `prompt_config` on a classification entry, a validation rule `pattern`, or has a `synthesize` state (a message names the key, never its value);
- `requires.handlers` names anything: a handler binding comes from a template, never from a model;
- the document is a `pattern` that carries a binding, a hook, variants, a generator, an options schema or a run field that names code or the driver channel (see the table above);
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

### The namespaces

Four [name registries](results_and_names.md#name-registries) hold the extension points, each with an entry-point group:

- `EXECUTABLE_KINDS` (`executable_kinds`, group `fsm_llm.executable_kinds`) maps a document `kind` to the adapter factory that turns a parsed envelope into an executable. `fsm`, `strategy` and `procedure` are registered, each to `FsmBodyExecutable`, and `fsm_llm.agents` registers `pattern` (to `PatternExecutable`) when it is imported. `adapter_for(kind)` raises `ExecutableKindError` for a kind with no adapter.
- `EXECUTABLE_SOURCES` (`executable_sources`, group `fsm_llm.executable_sources`) maps a source name to a function with no arguments that returns executables or envelope documents. `registry.load_sources()` calls each source that this registry has not called yet, once, and adds what it returns; a source or an item that fails is reported in the `ReloadReport` and warned, never raised. This is how `fsm_llm.agents` and `fsm_llm.reasoning` declare their built-ins without the engine importing them.
- `HANDLERS` (`handlers`, group `fsm_llm.handlers`) maps a handler name that a `pattern` file binds to its factory `factory(ctx, binding) -> list[Handler]`. The registry stores factories, never built handlers: a factory is called once per run with that run's `HandlerContext`, so per-run state is never in a registry. `fsm_llm.agents.handlers` registers the `agents.*` library when it is imported, so nothing is registered until a package imports its library. The factories by role: the tool loop (`agents.tool_executor`, `agents.think_turn_limiter`, `agents.iteration_limiter`), the keys a turn clears (`agents.think_fresh_keys`, `agents.approval_fresh_keys`, `agents.feedback_consumed` and the fresh-key handlers of single patterns), the approval gate (`agents.approval_gate`), the evaluators, reflectors, checkers, delegators and counters of the worker patterns (`reflexion`, `rewoo`, `plan_execute`, `maker_checker`, `debate`, `evaluator_optimizer`, `orchestrator`, `prompt_chain`), the function-calling group of `native_fc` (`agents.native_fc`) and the run hooks (`agents.verify_retry`, `agents.periodic_reflection`, `agents.memory_recall`, `agents.memory_remember`, `agents.raise_run_ending_error`, `agents.rewoo_cut_off_answer`). `HANDLERS.list()` is the list in force. A key a binding names is refused when it is internal-prefixed or secret-looking, and a binding cannot move the approval gate.
- `GENERATORS` (`generators`, group `fsm_llm.generators`) maps a generator name to a function that writes a pattern's FSM; `fsm_llm.agents` registers one, `agents.prompt_chain`.
- `PATTERN_AGENTS` (`pattern_agents`, no entry-point group) maps each built-in pattern id to the agent class `create_agent` builds for it; `fsm_llm.agents` fills it when it is imported and it is the one class table of the package. `builtin_executables()` follows its order.

A `pattern` document loaded through `ExecutableRegistry.add_document` resolves its handler and generator names through the last two namespaces, so a name that no package registered is refused; any other kind is not judged. The `fsm-llm-validate` command has no resolver and so refuses a pattern that names a handler.

A directory of pattern files, and `create_agent(id, registry=)` for a file a registry holds, are described in [Agents from pattern files](#agents-from-pattern-files).

The package `pyproject.toml` declares no entry point for either group yet, so a source is registered by an explicit `EXECUTABLE_SOURCES.register(name, factory)` call, and `load_sources()` finds only the packages the process has imported. An entry point installed after the process started is not seen until it restarts. `ExecutableRegistry(kinds=..., sources=...)` takes registries of its own.

The event vocabulary of the event log declares `run/start`, `run/end` and `executable/resolve` (`fsm_llm.runtime.EventType`). Only `settle` writes the first two, for a delegated `call` (see the `call` targets section below); nothing writes `executable/resolve`, a run through `run` adds no event of its own, and the registry logs nothing. Nothing reads them back, so a pin that a restart could rebuild from the log waits for that reader.

## Built-in executables

`fsm_llm.agents` and `fsm_llm.reasoning` declare their built-ins through the `executable_sources` namespace: each registers its lazy factories, named `agents`, `agent_presets` (the shipped presets) and `reasoning`, when the package is imported, and `ExecutableRegistry().load_sources()` calls the factories of the packages your process has imported (there is no entry-point wiring yet, so import the package first). The built-ins have lifecycle `builtin`: the agent patterns under their `create_agent` names (`react`, `debate`, and so on), the presets, the nine reasoning strategies (`reasoning.deductive`, and so on) and `reasoning.solve`.

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

A `PatternExecutable` (every pattern id, from a file or from code) and `reasoning.solve` are subclasses of `CodeBackedExecutable` (`fsm_llm.engine.executables`) that build their agent or engine per run and run it. The strategies are `strategy` documents on `FsmBodyExecutable`. Which pattern is a file and which is code:

- **Served from a file** under `fsm_llm/agents/patterns/`: `react`, `reflexion`, `rewoo`, `plan_execute`, `prompt_chain` (a generator), `native_fc`, `orchestrator`, `maker_checker`, `debate`, `evaluator_optimizer`, and `verified_react` and `auto_memory`, which extend `react` and add hooks and options. Their `create_agent` classes are thin subclasses of `PatternAgent`. The presets `code-review`, `summarize` and `data-extraction` extend `react` and have no class: a `PatternAgent` runs them.
- **Code-backed**: `self_consistency`, `adapt`, `swarm`, `parallel_react`, `meta_builder` and `reasoning_react`, and `reasoning.solve`. `parallel_react` stays code because its think prompt still carries the task text (see the known limits). `reasoning_react` is built on the reasoning engine, so the import edge from `agents` to `reasoning` stays. The files under `fsm_llm/agents/patterns/` and the `pattern_agents` namespace give the list in force; a file replaces a code-backed row under the same id.

- A pattern's input is `task` (and an optional `initial_context`). A pattern that needs Python objects (its tools, a chain, an evaluation function, maker and checker instructions, the members of a swarm) names them in `requires.callables` and is not `launchable`: the host supplies them in `RunContext(callables=...)`, a `NameRegistry`, and an unresolved name raises `ExecutableError` before any model call. Optional objects, such as `hitl` and `config`, are read from the same registry when present. Patterns that need none (`debate`, `self_consistency`, `adapt`, `plan_execute`, `orchestrator` and `meta_builder`, with `reasoning.solve`) are launchable.
- An executable never supplies an approval. The `hitl` object you give passes through unchanged, and a tool that needs approval without one is refused when the agent is built.
- `max_seconds` of the context becomes the agent's `timeout_seconds`, but never widens a host `config`: the smaller of the two is used (a host `config` of 30 seconds under a 600 second run budget keeps 30); `max_steps` is refused for a pattern (set `max_iterations` on the config) and both are refused by `reasoning.solve`, which has its own step budget.
- `meta_builder` is single-shot: `run(task)` returns the artifact in `outputs`, and a build that does not validate is `failed`, with the errors in `detail`.
- A strategy takes one input, `problem_statement`, declares no output, and reports both identities: a standalone run returns every non-None key of the strategy's `results` map, so a refused calculator expression returns its `calculation_error_details` instead of failing the run, and a caller with no `constraints` runs. A `call` state that targets a strategy and declares no `returns` merges the same map (every non-None mapped child value, a key the child never wrote simply absent); frame mode reads no contract, the call's own `inputs` are what the child receives. `simple_calculator` and `hybrid` run with the handler registration of the reasoning package as their `setup`. A pattern and `reasoning.solve` report the template identity only: their instance identity is `None`, because the tools of a run shape the instructions (a semantic tool registry retrieves its tools by the task) and the FSM is built per run.
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

- **What resolves.** The greatest version of a plain id of kind `fsm`, `strategy` or `procedure` that instantiates. `None` (the call ends `failed/unknown_target`) for an unknown id, a kind with no body to push (`pattern`, `workflow`), a body whose variable is unfilled (one warning naming the error type) and an id that only names an importable module (nothing is ever imported by id). Aliases and experiments are not routed by `resolve_target`. A pattern, and any executable whose `requires.handlers` is non-empty, is never pushed as a frame (a frame runs no handler), so a call that targets one ends `failed/unknown_target`; a built-in pattern runs in delegated mode. The one exemption is a built-in strategy with a host binding (`setup`, code only; the reasoning host registers its handlers on the `API` that pushes the frame): an operator strategy file that requires handlers is still refused.
- **What the child sees.** Only the `inputs` the call state declares, copied; `contract.inputs` is not read in frame mode. The child never receives the parent's context, an internal key or an approval grant. The parent receives only the declared `returns`, and for a `strategy` that declares none, every key of the strategy's own `results` map that the child wrote (`ResolvedTarget.returns`).
- **What the record carries.** The final `call_result` holds the template identity of the executable and the instance identity of what was pushed (the identities `run` would report for the same inputs), read from what `settle` pushed and never from the forgeable public record; after a restart that cannot prove it they are `None`.
- **Nested runs.** `FsmInstance.arun` gives its own `API` the context's registry (`RunContext.registry`), so a body with a `call` state runs under a run; a body with a call state and no registry is refused.
- **Delegated mode (a pattern as a target).** A pattern cannot be a frame: it needs its agent, which binds its handlers. With `API(targets=ExecutableTargets(registry, context))` (a resolver that is also a `RunLauncher`) the call runs it off the turn on a daemon thread, and a registry alone still ends such a call `failed/unknown_target`. Pass 1 reserves a minted handle and writes `call_result {status: running, mode: delegated, handle, deadline, ...}`; `settle` starts the run after the turn commits, from the parent's own state and only from that reservation, and writes `run/start` first (an fsynced checkpoint, `source: run`, the parent frame id); on every later turn the executor polls the host, and a finished run lands as ONE write (the final `call_result` and the declared `returns`, each reading a declared output of the run or the pseudo output `answer`, capped at `MAX_DELEGATED_RETURN_CHARS`). `aadvance` and the step streams (`advance_stream`, `aadvance_stream`, so also `run_until_terminal_stream`) make one bounded wait (`DELEGATED_WAIT_SECONDS`, 0.5) after their step so a run loop does not spin. The call's `max_seconds` defaults to 600 and is capped at 3600 (`DEFAULT_DELEGATED_MAX_SECONDS`, `MAX_DELEGATED_MAX_SECONDS`); past it the call ends `aborted/budget_spent` and the run is only detached. A handle the host never saw for that conversation (a restart, a restored snapshot, a fork, a forged `call_result`) ends `failed/run_lost` and nothing is re-run. The child receives only the declared `inputs`, never the parent's context or an approval grant. `run/end` is written by `settle` on the engine thread when the committed record is final, never by a worker. Only a pattern executable resolves (a built-in, a preset or a pattern file the registry holds), and only when the host context supplies its `requires.callables` (a `react` call needs a `tools` callable in `ExecutableTargets(registry, context).context`), it is not `conversational`, the call's `inputs` are inputs of its contract (`task`, and `initial_context` for most) and the call sets no `max_steps` for a pattern (a pattern refuses it): anything else ends `failed/unknown_target`. A call's `max_steps` reaches an executable that accepts a step budget (`reasoning.solve` receives it, narrowed against the host context's own `max_steps`). The child runs on the host context's interface, callables and event log, never on the parent's context, so two delegated runs share that interface on two worker threads (a sync-only interface must tolerate it; the usage meter of a `LiteLLMInterface` is locked and loses no call). A target of kind `workflow` is not built and ends `failed/unknown_target`: delegated workflows wait for workflow persistence and `Executable.resume` (roadmap E5.2 and E5.3).
- **A `procedure` target is not gated.** The call kind trusts the ids its author or its closed `choose` list names; it checks no provenance and no lifecycle but `deprecated`, which resolves to nothing and ends the call `failed/unknown_target` (the `synthesize` kind judges both, through its executor). Do not list a model-written procedure where an untrusted value can reach the list.

## Known limits

Each limit has the condition under which it applies.

- **The key names of `initial_context` are caller data in definition text.** The patterns that declare a `context_keys` option (`react`, `reflexion`, `rewoo`, `orchestrator`) list the string keys of the caller's `initial_context` in the field prompts of the instantiated FSM, so a key name is caller text in instruction text. Internal-prefixed names, run-output keys and `agent_trace` are left out (`caller_prompt_keys`), and the values never enter the definition. A key name is not otherwise sanitized: do not build `initial_context` keys from untrusted text.
- **No stall detection.** A body that needs a user message to leave a state (a transition gated on a value only a user supplies) runs message-free steps until the step budget is spent, and ends `aborted` with `budget_spent`. The run is bounded, never stuck. Give such a body its inputs as contract inputs, or do not run it as an executable.
- **Templates are bounded and pins are in memory.** The registry retains at most `MAX_RETAINED_TEMPLATES` templates beside the current ones. A template that was replaced, evicted and is not pinned cannot be found by its identity (`ExecutableEvictedError`), and a restart forgets every pin. A run that started and never ended is not pinned from the log: a delegated `call` writes `run/start` and `run/end`, but no reader turns them into a pin yet.
- **Cancel hazard of the patterns.** A pattern and `reasoning.solve` run their synchronous body on a worker thread (`aoffload`). A cancelled `await` returns at once, but the worker keeps running to its end: its tools and handlers are not stopped, and its late result is dropped. The cancel token is checked only before the run starts. The same holds for an `AgentStep` whose agent has no `arun` of its own, and for the default `BaseAgent.arun`: it is one worker hop, not an async body.
- **A pattern has no instance identity.** A pattern served from a file has the digest of its document as template identity (a preset the digest of its flat document), and a code-backed one the digest of its code claim. Both report `RunResult.identity` with the template identity and `instance_identity=None`, because the per-run tools shape the instructions (and a semantic tool registry retrieves its tools by the task). A caller's task never enters either identity or the FSM description: every agent builder writes a static description and the task is the input `task`. Exceptions keep task-derived text in definition text: `parallel_react` writes the task into its think prompt line, and a `SemanticToolRegistry` makes the tool section of `react`, `reflexion`, `rewoo` and `plan_execute` depend on the task.
- **A delegated run is not interruptible and not resumable.** A deadline, a cancel, `end_conversation` and `close()` only set the run's token. A pattern checks it once, before the run starts, never between steps (`CodeBackedInstance.arun`, and `PatternExecutable` does not pass it to its agent), so a cancel after the start only detaches the run: a detached pattern may keep running and calling tools on its daemon thread, and nothing joins it. After a restart, a snapshot restore or a fork the run is lost (`failed/run_lost`) and nothing is re-run, because the host scopes a handle to the conversation that reserved it: even a fork on the same `API`, taken while the run was in flight, adopts nothing; a fork taken after the final record is a normal fork. The mark that owes `run/end` is in memory, so after a crash a log with a `run/start` and no `run/end` stays that way, even after the resumed process wrote the final `failed/run_lost` record, and so does a log whose process died between the turn that committed the final record and the append of `run/end`; a reader takes a `run/start` with no `run/end` as lost. A host context that sets `max_steps` makes every pattern fail (a pattern refuses it).
- **A detached pattern delays the exit of the process.** `PatternExecutable` runs its agent on an executor thread of the run's own event loop, and the interpreter joins executor threads at exit, so a process that exits while a pattern is still running its agent waits for what is left of that run, bounded by the agent's own timeout (the call's `max_seconds`, 600 by default, 3600 at most). A run that only awaits is killed with its daemon thread. The parent's log stays intact either way: the run thread writes no run event into it (a pattern handed the host's event log writes the files of its own agent conversations).
- **Finished results are consumed, bounded and few.** The host keeps the result of a finished run only until the record that lands it is committed (`settle` then drops it with the optional `forget` method of the launcher) or its conversation ends, and caps what it keeps at `MAX_FINISHED_RUNS` (256, owner-visible): results whose parents have not polled yet count toward it, the oldest goes, and its parent ends `failed/run_lost` for work that succeeded, so a host with more than 256 unpolled finished runs at once loses the oldest. Each stored result is capped when the run ends, not only when it is merged: every value of `outputs` and `detail` and the `answer` follow the 8000 character rule (a longer text is cut at 8000 characters and ends with a marker, so it is 8000 characters plus the marker; a larger container becomes a one-line marker). A failed `run/start` write (a full disk, a log that cannot be written) puts the reservation back: the end settle of the turn entry logs it at WARNING and the next entry starts the same handle, until the call's own deadline.
- **A delegated run's neighbours: the approval callback, the executor pool, the clock.** The host context's `hitl` object reaches a delegated `react` unchanged and its approval callback runs on the run's worker thread, with no user attached, and two concurrent runs share one `HumanInTheLoop`: it must be thread-safe and must not block on a console (`input()` parks the thread until the agent's timeout). The 0.5 second wait of `aadvance` and the step streams takes one slot of the event loop's default executor (`asyncio.to_thread`, `min(32, cpu + 4)` threads) for up to 0.5 seconds, so a host with that many delegated parents in async run loops saturates it and the sync-only twins of an interface queue behind the waits: size the executor, or keep the number of concurrent delegated run loops below it. A cancel during that wait (a task cancel) is not swallowed and drops the `AdvanceResult` of a step that is already committed (the end settle had the same window before); a launcher whose `poll` or `wait` raises costs one WARNING and never turns a committed step into an error. The `deadline` of a call is wall-clock (`time.time()`, so it survives a snapshot): a forward jump of the system clock (NTP, a suspend) can abort every running delegated call at once as `aborted/budget_spent`.
- **The identities of a delegated run come from the host.** The final record and `run/end` name the template and instance the host STARTED the run with (`RunPoll.template_identity`), so a registry reload during the run does not change them and a forged `call_result` cannot name them; a run that never started (lost) is described by the current registry. A host that cannot start a run (full at start, a thread the process cannot start) ends the call `failed/push_failed` at once and writes no `run/start`; a failure while settle prepares the run (resolution, copying an input) ends it `failed/push_failed` at once too, naming only the exception type in the log, instead of waiting for the deadline. `is_run_launcher` reads the six methods on the CLASS of the resolver: a `MagicMock` is no launcher (use a real class for a double), and a proxy that forwards names through `__getattr__` is none either.
- **A popped frame stops its runs.** `pop_fsm` (a hand pop, or the engine's pop of a child whose budget is spent) tells the launcher to stop the runs of the popped frame and writes `run/end` (`aborted`, `cancelled`) right after `stack/pop`; a detached pattern may keep running as above. `end_conversation` drops the runs of the ended conversation without a `run/end` (its log writer is released), so a `run/start` with no `run/end` reads as lost.
- **A delegated parent spends a step per poll.** Every turn of a parent in a delegated call state polls once and is a step, and `aadvance` and the step streams wait at most 0.5 seconds for the run (one helper, `wait_for_run`, one semantics), so the `max_steps` of a run loop over it must cover the seconds of the run divided by 0.5. Leave the call state silent (no `response_instructions`) or every poll turn calls the model. `aconverse` and the message streams never wait.
- **Budgets are per run.** `RunContext.derive` narrows a limit but does not share a ledger: the steps a child spends are not taken from its parent's budget.
- **No entry-point wiring.** `pyproject.toml` declares no entry point for `executable_kinds` or `executable_sources`, so a source is found only when its package was imported and registered it.
- **No launch by id from the monitor.** The monitor still starts agents from its own table of classes and workflows from its own presets; it does not list or launch executables, and a pattern with `requires.callables` could not be launched there.
- **The meta-builder writes no envelope.** `MetaBuilderResult.artifact` and `fsm-llm-meta -o` stay the bare FSM, workflow or agent definition; `--executables DIR` only adds the ids of your pattern files to the valid `agent_type` values, and the monitor has no such flag.
- **No `resume`, no `target` on `AgentStep` or `ConversationStep`.** An `ExecutableStep` is the only workflow step that runs an executable by id.
- **The greeting is outside `max_seconds`.** An FSM-body run starts its conversation (the greeting, the first model call when the initial state speaks) before the message-free step loop, and `max_seconds` bounds only that loop, so a slow greeting adds to it. Use the step's `timeout_seconds` (which cancels the whole run) when the total matters.
- **`ExecutableStep` does not set `RunContext.parent`.** The run is one the step starts; no parent run reference is passed, so a derived child budget or a log link to the workflow instance is not built.
- **A pattern or built-in is checked at construction only when the registry serves it.** A target a registry has not loaded yet is judged when the step runs.
- **Importing `fsm_llm.agents` or `fsm_llm.reasoning` loads the `fsm_llm.engine.executables` package** (with the run context and the document loader), because each registers its source in the namespace that lives there; the executables of the two packages load only when a registry loads the sources.
- **Claims are not authorization.** `lifecycle`, `capabilities`, `contract` and `requires` describe a document; the registry serves a `trusted` or `builtin` one exactly as it serves a `draft`.

## Next steps

- [Runs, results and names](results_and_names.md) for the `RunResult` that carries both identities and for `NameRegistry`.
- [FSM Design Guide](fsm_design.md) for the FSM inside `body.fsm`.
- [Runtime FSM synthesis](synthesis.md) for the library that stores model-written FSMs by digest.
- [Command line](cli.md) and the [API Reference](api_reference.md).
