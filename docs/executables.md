# Executables

> Covers FSM-LLM v0.13.0

An executable is one JSON document that wraps an FSM body with a header: who it is, what it reads and writes, which names it needs, and which `{{variables}}` an operator fills in. This page covers the format, the two identities of a document, the rule that keeps a caller's input out of instruction text, the load sequence that checks a document, and how `fsm-llm-validate` reports it.

What exists today is the format and its loader. A document is parsed and checked, and `instantiate_envelope` turns it into an `FSMDefinition` that you hand to `API.from_definition`. The engine does not read an executable file by itself and nothing here executes anything from a document. The built-in agent patterns, the reasoning strategies and `reasoning.solve` are served by id as executables (see [Built-in executables](#built-in-executables)).

## The document

```json
{
  "schema": "fsm-llm/executable",
  "schema_version": "1.0",
  "kind": "fsm",
  "id": "support/greeter",
  "version": "1",
  "name": "Greeter",
  "description": "Ask a name, say goodbye",
  "contract": {"inputs": [], "outputs": ["name"]},
  "capabilities": ["conversational"],
  "variables": ["company"],
  "body": {
    "fsm": {
      "name": "Greeter",
      "description": "Ask a name, say goodbye",
      "initial_state": "ask_name",
      "persona": "A receptionist at {{company}}",
      "states": {
        "ask_name": {
          "id": "ask_name",
          "description": "Find out the name",
          "purpose": "Learn the user's name",
          "extraction_instructions": "Extract the user's name",
          "response_instructions": "Ask for the user's name",
          "required_context_keys": ["name"],
          "transitions": [
            {
              "target_state": "goodbye",
              "description": "The name is known",
              "conditions": [
                {
                  "description": "A name was collected",
                  "requires_context_keys": ["name"],
                  "logic": {"has_context": "name"}
                }
              ]
            }
          ]
        },
        "goodbye": {
          "id": "goodbye",
          "description": "Say goodbye",
          "purpose": "Wrap up the conversation",
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
| `kind` | `fsm`, `strategy` or `procedure`. It decides the shape of `body`. The kinds `pattern` and `workflow` are refused by name until their adapters exist. |
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

## Built-in executables

`fsm_llm.agents` and `fsm_llm.reasoning` declare their built-ins through the `executable_sources` namespace: each registers one lazy factory, named `agents` and `reasoning`, when the package is imported, and `ExecutableRegistry().load_sources()` calls the factories of the packages your process has imported (there is no entry-point wiring yet, so import the package first). The built-ins have lifecycle `builtin`: the 18 agent patterns under their `create_agent` names (`react`, `debate`, and so on), nine strategies (`reasoning.deductive`, and so on) and `reasoning.solve`.

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

- A pattern's input is `task` (and an optional `initial_context`). A pattern that needs Python objects (its tools, a chain, an evaluation function, maker and checker instructions, the members of a swarm) names them in `requires.callables` and is not `launchable`: the host supplies them in `RunContext(callables=...)`, a `NameRegistry` of that name, and an unresolved name raises `ExecutableError` before any model call. Optional objects, such as `hitl` and `config`, are read from the same registry when present.
- An executable never supplies an approval. The `hitl` object you give passes through unchanged, and a tool that needs approval without one is refused when the agent is built.
- `max_seconds` of the context becomes the agent's `timeout_seconds`; `max_steps` is refused for a pattern (set `max_iterations` on the config) and both are refused by `reasoning.solve`, which has its own step budget.
- A strategy takes the four keys the reasoning engine hands it (`problem_statement`, `problem_components`, `constraints`, `problem_type`) as inputs and reports both identities. A pattern and `reasoning.solve` report the template identity only: their instance identity is `None`, because the body is code and the task sits in the FSM description until patterns become files.
- Known limits: a pattern or `reasoning.solve` runs its synchronous body on a worker thread, checks the cancel token only before it starts, and leaves the worker running if its await is cancelled.

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

The `resolver` is a function called with `(category, name)`, the category being `tools`, `handlers`, `callables` or `executables`. The loader reads no registry itself; build the function from your [name registries](results_and_names.md#name-registries): `lambda category, name: name in namespaces[category]`. No namespace exists yet for executable sources, so the `fsm-llm-validate` command passes no resolver.

### Text written by a model (`origin="model"`)

`origin` says who wrote the document. `"operator"` is a file a person wrote and V7 never applies. `"model"` is text a model produced, and V7 then refuses it when:

- a `requires` name of `tools`, `callables` or `executables` is not on the `allow` list (`allow` maps a category to the names a model may require; `None` is an empty list, so every name is refused);
- a tool that a completion state of the body declares is not on the `tools` list. The body is read as written and, when variables were filled, as filled, and the check runs whatever `requires.tools` says: the claim `"some"` cannot be checked and is refused, the claim `"none"` beside declared tools is a disagreement and is refused, and a list that omits a declared tool is a `[names]` error as well;
- the body sets a `persona`, a `model` or `prompt_config` on a classification entry, a validation rule `pattern`, or has a `synthesize` state (a message names the key, never its value);
- `requires.handlers` names anything: a handler binding comes from a template, never from a model;
- `lifecycle` is anything but `draft`.

A body key that would bind code (`handlers`, `variants`, `generator`) is refused by V1 for every origin. Names, keys and state ids that came from a document reach a message only after they are capped and stripped of control characters, because a repair loop may show the message to a model. V7 has no consumer in the library yet. It is not the synthesis gate G5: V7 covers the lines above (the `requires` names, the declared tools, those four body settings, handlers and lifecycle) and nothing else of G5, so text length, key names, unknown keys, priorities and the abort edge are still checked only by G5, and a synthesis or meta-builder output needs both.

### Known limits

- `newer_envelope_fields(envelope)` lists fields newer than the declared `schema_version`, but `load_executable` does not call it. Only version `1.0` exists, so there is nothing newer to report yet.
- The contract check (V5) warns and never refuses.
- Claims are not authorization, as above.

## `fsm-llm-validate` and executables

```bash
fsm-llm-validate --fsm greeter.exe.json
```

The command reads the file and routes on its header: a file whose `schema` is `fsm-llm/executable` is validated as an executable, anything else as an FSM definition, whatever the extension. The report is the same JSON as for an FSM (`fsm_name` is the document `id`, then `is_valid`, `errors`, `warnings`, `info`), with the step tags in each line. It checks the template as written (no variables given) and resolves no names. The exit code is 0 for a valid document and 1 otherwise.

`--upgrade` rewrites an older FSM definition file. An executable is never rewritten: with `--upgrade` the command prints the report, logs one warning and exits 1, and no file changes.

## Next steps

- [Runs, results and names](results_and_names.md) for the `RunResult` that carries both identities.
- [FSM Design Guide](fsm_design.md) for the FSM inside `body.fsm`.
- [Runtime FSM synthesis](synthesis.md) for the library that stores model-written FSMs by digest.
- [Command line](cli.md) and the [API Reference](api_reference.md).
