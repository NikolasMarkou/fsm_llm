# 2026 restructuring: one executable format for agents, reasoning, synthesis and workflows

Status: proposal, not started. Written 2026-11-19 against v0.13.0. It turns a draft (a `procedure/` substrate and JSON definitions for every package) into a plan aimed at eleven targets. Every claim about today's code was checked against `src/fsm_llm/`. Where the draft conflicts with the [2026 roadmap](2026_roadmap.md), Section 8 names the roadmap change it needs.

## 1. Targets

| # | Target | Delivered by |
|---|---|---|
| T0 | One unified JSON schema for every executable: FSMs, agent patterns, reasoning strategies, synthesized procedures and workflows | Section 4.1, phase R1 |
| T1 | User-defined patterns without code changes: a `my_pattern.json` in a pattern directory makes `create_agent("my_pattern")` work | Sections 4.3 and 4.5, phase R4 |
| T2 | Hot-reloadable definitions: edit a file, reload the registry, new conversations use the new definition, no restart | Section 4.3, phase R2 |
| T3 | Cross-package interop: a workflow step runs a reasoning strategy, an agent pattern or a synthesized procedure, and its outputs merge back | Sections 4.2, 4.4 and 4.8, phases R3 and R5 |
| T4 | Dynamic strategy selection: reasoning's classifier can choose any registered executable that fits the strategy contract | Section 4.6, phase R3 |
| T5 | Synthesis compiles to JSON targets: `compile(goal, target="pattern:react")` and `target="workflow"` | Section 4.9, phase R6 |
| T6 | Gates and policy as JSON data: operators edit the allow-list and the gate pipeline without Python | Section 4.10, phase R6 |
| T7 | Async execution everywhere: parallel agents, streaming reasoning, concurrent simulation trials | Section 4.11, phase R7 |
| T8 | Checkpoint and resume of a running executable (context and stack) | Section 4.12, phases R3 and R5 |
| T9 | A/B testing: the registry serves `react_v1` or `react_v2` behind one name | Section 4.3, phase R2 |
| T10 | Validation up front: every definition is validated against the schema, and every name it uses is resolved, at load time, before any model call | Section 4.13, phase R1 |

## 2. Today, as verified

| Package | Driver | Child frames | Selection | Definitions |
|---|---|---|---|---|
| agents | `BaseAgent._standard_run` (`agents/base/runs.py`) on `API.run_until_terminal` | none of their own; `ReasoningReactAgent` reaches reasoning (the permanent `agents -> reasoning` edge) | `_PATTERNS` dict in `agents/__init__.py` | Python builders `build_<x>_fsm(...)` in `agents/fsm_definitions/`, parametrised by tools, flags and task text; handlers attached in Python (`agents/handlers/`) |
| reasoning | `ReasoningEngine.solve_problem` on `run_until_terminal(..., before_step=_StrategyStack)` | `_StrategyStack` pushes a strategy FSM, pops it when it ends, force-pops after `Defaults.MAX_SUB_FSM_STEPS`, merges through `merge_reasoning_results` | a classifier FSM, then the `ReasoningType` enum | module-level dicts in `reasoning/reasoning_modes/`, static apart from interpolated constants; `StrategySpec` and `STRATEGIES` in `reasoning/constants.py` hold aliases, result mapping and solution key |
| synthesis | `ProcedureRunner.run` on `run_until_terminal(..., before_step=hook)` | `ProcedureRunner` pushes the compiled child with declared inputs, bounds it with `push_fsm(max_steps=policy.max_turns)`, merges return keys on success | library search by goal | `FSMDefinition` JSON stored content-addressed by `FSMLibrary`; `SynthesisPolicy` is a frozen, validated dataclass |
| workflows | own async loop (`workflows/engine/driver.py`) | none: `ConversationStep` builds a fresh `API` and loops `converse`; `AgentStep` calls `agent.run(task)` | `WorkflowDefinition` by id | Pydantic step models holding Python callables; `serialize()` writes a dict with type tags, nothing reads it back |

What core already provides and this plan builds on: the step driver (`advance`, `run_until_terminal`, `run_until_terminal_stream`), per-frame budgets (`push_fsm(max_steps=, max_seconds=)`), the state-kind registry (`core/state_kinds.py` `KindSpec`, `engine/kinds/` `StateExecutor`), the definition cache keyed by `fsm_id` (a changed definition is a new cache entry, which makes hot reload safe), and `API.fork` and `API.resume` on the event log. What it does not provide yet: async `API` methods (roadmap 1.5) and entry-point plugins (roadmap 5.5).

The draft's assumptions that the code refutes, and what replaces them:

| Draft assumption | Code | Plan |
|---|---|---|
| Four separate substrates need one new runner | Three already run on core's step driver | The shared driver is a state kind (`call`) plus an executable resolver, both in the engine |
| Two budget implementations | Core frame budgets shipped; only reasoning keeps a private forced pop | Delete it in R3 |
| Workflow steps re-implement push and pop | They push nothing | Interop is a step that resolves an executable (R5) |
| Checkpoint and resume must be built | `fork` and `resume` exist | Move driver state into frames so resume covers it (R3) |
| Every pattern is a static JSON file | `prompt_chain` generates states; `react` adds states by flag | Variants as JSON patches; a generated topology names a registered generator (Section 4.1) |
| Gates can skip or repair | Gate order is fixed by an anchor in `synthesis/gates.py`; G5 is the security allow-list | The pipeline file can add and configure gates, never drop or reorder the mandatory ones (Section 4.10) |

## 3. Rules

1. `FSMDefinition` stays the one FSM program. The unified schema is an envelope around it, not a second FSM format; every FSM inside an executable is an `FSMDefinition` that loads exactly as `API` loads it.
2. A new state behaviour is a new state kind with an executor and verifier hooks (C2), never a branch in the turn pipeline.
3. No parallel runner, loader or tool registry. The envelope loader calls the existing loaders; agents run on `_standard_run`; workflows run on their engine.
4. JSON never carries code. A definition names handlers, tools, callables, generators and executables; each name resolves to code registered by Python. "Without code changes" (T1) means any combination of registered building blocks; a new building block is still Python.
5. Security boundaries do not move into JSON. The HITL grant (`_approval_granted` bound to one call) is enforced in the tool executor whatever the pattern file says; G1 to G5 and G6a always run; OKF, skills and procedure text keep their untrusted labels.
6. Model-written definitions (T5) may only name handlers, tools, callables and executables from the policy allow-list. Files on disk are trusted operator configuration, held as private as the session directory.
7. Registries are values. Each `API`, agent factory, engine or synthesizer takes the registry it uses; the module default is built once and can be frozen.
8. No shims. Moves are recorded in `tests/fixtures/module_path_map.json`; paths that `examples/` imports stay packages that re-export their names (`fsm_llm.agents.fsm_definitions`, `fsm_llm.agents.definitions`, `fsm_llm.agents.tools`, `fsm_llm.agents.python_tools`, and the public names of `fsm_llm.agents`, `fsm_llm.reasoning`, `fsm_llm.workflows`).

## 4. Design

### 4.1 The unified executable schema (T0)

Every executable is one JSON document with a common header and a body discriminated by `kind`:

```json
{
  "schema": "fsm-llm/executable",
  "schema_version": "1.0",
  "kind": "pattern",
  "id": "react",
  "version": "2",
  "name": "ReAct",
  "description": "Think, act with a tool, observe, conclude",
  "contract": {
    "inputs": ["task"],
    "outputs": ["final_answer"],
    "handler_only_keys": []
  },
  "requires": {
    "tools": "some",
    "handlers": ["agents.tool_executor", "agents.think_turn_limiter"],
    "callables": [],
    "executables": []
  },
  "capabilities": ["tools", "hitl"],
  "lifecycle": "builtin",
  "variables": ["task_description"],
  "body": {},
  "metadata": {}
}
```

| Field | Meaning |
|---|---|
| `schema`, `schema_version` | the envelope format and its version, registered like `FIELD_VERSIONS` so a newer field in an older document warns or refuses exactly as FSM fields do |
| `kind` | `fsm`, `strategy`, `pattern`, `procedure` or `workflow` |
| `id`, `version` | the registry name and the author's version label; the content hash (the `fsm_id` for FSM kinds) is the identity the log records |
| `contract` | the keys the executable reads and writes; checked at load against the data-flow analysis (Section 4.13), so it cannot drift from the body |
| `requires` | every name the body uses, resolved at load |
| `capabilities` | closed vocabulary: `tools`, `hitl`, `streaming`, `strategy` (selectable by reasoning), `conversational` (talks to a user) |
| `lifecycle` | `builtin`, `draft`, `verified`, `trusted`, `deprecated` (the synthesis vocabulary, extended by `builtin`) |
| `variables` | text placeholders `{{name}}` filled at instantiation; an unresolved variable fails (the rule of roadmap 2.3) |

Body by kind:

| Kind | Body | Today's source |
|---|---|---|
| `fsm` | `{"fsm": FSMDefinition}` | FSM JSON files |
| `strategy` | `{"fsm": FSMDefinition, "strategy": {"aliases": [...], "results": {...}, "solution_key": "..."}}` | `reasoning_modes/*.py` dicts and `StrategySpec` |
| `pattern` | `{"fsm": FSMDefinition, "handlers": [{"handler": "...", "timing": "...", "states": [...], "priority": 0}], "answer_key": "...", "options": {...}, "variants": {"name": [patch ops]}}`, or `{"generator": "agents.prompt_chain", "options": {...}}` for a generated topology | `agents/fsm_definitions/*.py` and the agent classes |
| `procedure` | `{"fsm": FSMDefinition, "provenance": {...}}` | synthesis library entries |
| `workflow` | `{"initial_step_id": "...", "steps": {...}}` in the form `WorkflowDefinition.serialize()` writes, with callables as registered names | `WorkflowDefinition` objects |

`variants` are RFC 6902 patch lists, selected by options at instantiation (`react` with `approval: true` applies the `approval` variant). The patch applier moves down from `synthesis/patch.py` to `core/definitions/`, where both synthesis repair and pattern variants use it. `generator` is the escape hatch for topologies that depend on input (one state per chain step): the name resolves to a registered Python function whose output must be an `FSMDefinition` and is validated like any body.

The JSON Schema file is generated from the Pydantic models (a discriminated union on `kind`), shipped as package data, and held equal to the models by a test. It is never hand-edited. State-level mixing of workflow steps and FSM states in one graph (the draft's `hybrid` kind) is replaced by calls across kinds (Section 4.4): an FSM state calls a workflow, a workflow step calls an FSM.

### 4.2 The executable protocol and resolver

The engine defines what every kind can do, so callers never branch on the package:

| Member | Meaning |
|---|---|
| `id`, `kind`, `contract`, `identity` | from the envelope; `identity` is the content hash |
| `instantiate(options, variables) -> Executable` | applies variants and variables, validates the result |
| `run(inputs, *, budget, llm_interface) -> ExecutableResult` | runs to the end; `ExecutableResult` holds a C8 `status`, `reason`, `outputs`, `answer` and the run handle |
| `arun(...)` | the async twin, after roadmap 1.5 |
| `resume(handle)` | continues a run after a crash (Section 4.12) |

Kind adapters implement it: the engine owns `fsm`, `strategy` and `procedure` (all FSM bodies); agents own `pattern`; workflows own `workflow`. The engine may not import a subpackage, so the subpackage adapters are found through one entry-point group, `fsm_llm.executable_kinds`, declared in `pyproject.toml` and loaded on first use of that kind (this pulls one group of roadmap 5.5 forward). `import fsm_llm` loads no adapter; the laziness test stays as it is.

### 4.3 The registry (T1, T2, T9)

`ExecutableRegistry` in the engine:

| Feature | Behaviour |
|---|---|
| Sources | built-ins shipped as package data (`agents/patterns/*.json`, `reasoning/strategies/*.json`), user directories (`ExecutableRegistry.from_paths([...])`, and `--executables DIR` on the CLIs), in-memory documents, the synthesis library |
| Lookup | `get(id)` the current version; `get(id, version=...)`; `list(kind=..., capability=...)`; `metadata(id)` |
| Reload (T2) | `reload()` re-reads every source, validates everything first, then swaps atomically; a file that fails is reported and its previous version stays served |
| Retention | every identity ever served stays resolvable by hash for the process lifetime, and on disk when a retention directory is set, so a running or resumed conversation keeps its definition |
| Aliases and variants (T9) | an alias entry `{"id": "react", "route": {"react_v1": 0.5, "react_v2": 0.5}, "key": "conversation"}` assigns deterministically by a hash of the experiment name and the conversation id; the chosen identity is logged (`executable/resolve` event), so eval groups results by identity |
| Collisions | a duplicate id from two sources is refused unless a source is declared as an override |

User directories are the place for T1: dropping files into the installed package directory is not supported (it is read-only in many installs and lost on upgrade).

### 4.4 The `call` state kind (T3, T8)

A state with a `call` config runs any registered executable from inside an FSM:

| Field | Meaning |
|---|---|
| `target` | an executable id; or `target_key` plus `targets`: a handler-only key naming the target at run time and the closed list it may hold |
| `inputs` | parent keys passed in; nothing else is |
| `returns` | `{parent_key: child_output}`; copied on success, only non-`None` values |
| `on_failure` | the state an edge must reach for `failed` and `aborted` (the loader rule `synthesize` has today) |
| `result_key` | default `call_result`, handler-only, `{status, target, identity, reason}` |
| `max_steps`, `max_seconds` | the child's budget |

Two modes, chosen by the target's kind:

- Frame mode for FSM bodies (`fsm`, `strategy`, `procedure`): the child is pushed as a stack frame, driven by core's step driver, popped at its end. It is in the event log and resumes with the conversation.
- Delegated mode for `pattern` and `workflow`: the adapter starts a child run (an agent conversation or a workflow instance) whose handle is written to the log before it starts; the parent state waits on it, and a resumed parent resumes the child by its handle.

`synthesize` becomes a `call` whose target comes from the synthesizer and library (roadmap P7.1); the two share the executor body and `synthesize` keeps its fields and `synthesis_result`, which are wire contracts. The executor needs a way to request a push or pop that the turn's stack module applies, so the lock order, the rollback snapshot and the `stack/push` and `stack/pop` events stay where they are; that is the main engineering risk (Section 10).

### 4.5 Agents from pattern files (T1)

- A handler library in agents registers the existing handlers by name: `agents.tool_executor` (`AgentHandlers`), `agents.think_turn_limiter`, `agents.forced_conclusion`, `agents.approval_gate`, `agents.trace_recorder`, `agents.run_guard` and the others in `agents/handlers/`. Each name maps to a factory that receives the run's tools, config and HITL objects.
- `PatternAgent` runs any `pattern` executable on `_standard_run`, attaching the handlers its file binds. `create_agent(pattern, tools=..., registry=...)` resolves the id and returns a `PatternAgent` or the built-in class registered for that id.
- Built-in patterns become pattern files; `ReactAgent` and the other classes keep their names and become thin subclasses bound to their id, keeping only logic that is not yet a named handler. `VerifiedReactAgent` and `AutoMemoryReactAgent` become presets over three hook points (`before_run`, `on_iteration`, `after_run`), which a pattern file may also bind by name.
- HITL: a pattern whose tools include an approval-required tool must bind `agents.approval_gate` or it is refused at creation; the grant check itself stays in the tool executor, so no file can skip it.
- `fsm_definitions/` stays a package re-exporting `build_<x>_fsm`; each becomes `instantiate` on the pattern file and returns the same dict.

### 4.6 Reasoning from strategy files (T4)

- Every strategy, the classifier and the orchestrator become `strategy` and `fsm` files under `reasoning/strategies/`. `StrategySpec` becomes the `strategy` body; `STRATEGIES`, `ALL_REASONING_FSMS`, `map_reasoning_type`, `get_available_reasoning_types` and the CLI `--type` choices read the registry.
- The orchestrator's push becomes a `call` state with `target_key` and `targets`; `_StrategyStack`, `MAX_SUB_FSM_STEPS` and `merge_reasoning_results` are deleted. The analytical fallback (the D-009 anchor in `reasoning/engine.py`) stays in the handler that writes the selected type.
- Selection over any executable: at the start of a solve the engine lists registry entries with the `strategy` capability whose contract reads only the problem keys and writes a solution key, builds the classifier's intents from them (id and description), and passes the same list as `targets`, so the model can choose only among them. A user strategy or an agent pattern declared `strategy`-capable is selectable without code.
- `ReasoningType` stays the vocabulary of the built-ins (its values are wire strings); functions that return a strategy return `str` for others.

### 4.7 Synthesis library as executables

Library entries are `procedure` documents (the stored `FSMDefinition` inside the envelope, provenance and lifecycle in the header). The registry reads a library as a source, so a verified procedure is callable by id from a `call` state, a workflow step or the reasoning classifier. The storage itself moves to the runtime store with roadmap P7.2.

### 4.8 Workflows as executables (T3)

- The `workflow` body is the form `serialize()` writes. A loader reads it back; every callable is a name in a callable registry; `load(serialize(w)) == w` for every step type.
- A new `ExecutableStep` (`{"type": "executable", "target": "deductive", "inputs": {...}, "returns": {...}, "success_step_id": "...", "error_step_id": "..."}`) runs any registered executable through the resolver and maps its outputs back into the workflow context. `AgentStep` and `ConversationStep` keep working and gain a `target` by name.
- Workflows keep their own engine: parallel branches, timers and external events have no FSM counterpart. A workflow is called from an FSM through `call` in delegated mode.
- A sync caller uses `fsm_llm.aio.run_sync`.

### 4.9 Compile targets (T5)

| Target | What the model writes | What it may not write | Gates |
|---|---|---|---|
| `fsm` (today) | an `FSMDefinition` | settings, persona, regex, model names (G5) | G1 to G7 as today |
| `pattern:<id>` | the FSM body of a pattern, inside the slots the template pattern declares (state ids with roles, required transitions) | handler bindings, variants, options: copied from the template, never generated | G1 to G7 on the FSM, plus a role check that every bound handler still finds its state |
| `workflow` | a `workflow` body whose steps are `executable` steps and routing steps | callables outside the policy allow-list, any `api` or Python step | W1 loads, W2 structure (no dangling step ids, every step reachable, an end reachable from every step, through `dependency_resolver`), W5 policy, W6 dry run with stubbed executables; no G6b until the simulator can drive a workflow, so a compiled workflow tops out at `draft` unless a human approves it |

All compiled targets are stored as executables in the library and become callable by id once their lifecycle allows it.

### 4.10 Gates and policy as data (T6)

- `SynthesisPolicy.from_json(path)` and `to_json()`: the dataclass already validates at construction. Hard ceilings stay in code (`synthesis/constants.py`); a file may tighten any limit and may not exceed a ceiling.
- A gate pipeline file: `{"gates": ["G1", "G2", "G3", "G4", "G5", "G6a", "host:no_network_tools", "G6b", "G7"], "params": {...}}`. G1 to G5 and G6a must all be present, in that order; host gates (registered Python callables, by name) may be inserted after G6a; G6b and G7 are configured by policy as today. A file that drops or reorders a mandatory gate is refused at load.
- These files are operator configuration: loaded from paths confined with `security.resolve_within`, never written by a model.

### 4.11 Async (T7)

`Executable.arun` on every adapter after roadmap 1.5 (`API.aconverse`, `aadvance`, `arun_until_terminal`). Then: `ParallelReactAgent` and the swarm on asyncio tasks instead of a thread pool; workflow `executable` steps awaited natively; G6b trials concurrent under admission control (roadmap 1.6), so a model server with one slot is not flooded. Streaming reasoning does not wait: `run_until_terminal_stream` exists today, and R3 gives `solve_problem` a streaming form on it.

### 4.12 Checkpoint and resume (T8)

Built on the event log, not a second checkpoint format. Frame-mode calls are frames, so `API.resume` restores them; a frame is trusted only through a definition whose recomputed identity matches, which the registry's retention guarantees. Delegated runs write their handle before they start; `Executable.resume(handle)` continues an agent conversation through `API.resume` and a workflow instance through workflow persistence (roadmap 1.8). Agent runs keep their run state in context or in the log, not in Python attributes, so a resumed run behaves like an uninterrupted one.

### 4.13 Validation up front (T10)

A document passes, in order, before any model call:

1. JSON Schema (the generated file) and the Pydantic models;
2. the body loader of its kind: `FSMDefinition` load and `FSMValidator` for FSM bodies, the workflow loader and structure checks for workflows;
3. name resolution: every handler, tool, callable, generator and executable in `requires` resolves, and every name the body uses is in `requires`;
4. variants and variables: every variant applies to the base and the result validates; every placeholder is declared;
5. contract check: declared inputs and outputs agree with `core.verify.flow` on the FSM body;
6. optional strict mode for registry loads: G3, G4 and G6a.

`fsm-llm-validate` accepts executable documents and reports every failure of a file at once. A registry load reports per file and keeps serving the rest.

### 4.14 Layer placement

| Piece | Where |
|---|---|
| envelope models, generated JSON Schema, schema version table, patch applier, `call` KindSpec and verifier hooks | `core/` |
| `Executable` protocol, `ExecutableRegistry`, resolver, FSM-kind adapters, `call` executor, the `fsm_llm.executable_kinds` entry-point loader | `engine/` |
| handler library, `PatternAgent`, pattern adapter, pattern files | `agents/` |
| strategy files, classifier over the registry | `reasoning/` |
| workflow loader, callable registry, `ExecutableStep`, workflow adapter | `workflows/` |
| compile targets, policy and gate pipeline files, library as a registry source | `synthesis/` |

No new layer package. The subpackages still do not import each other: cross-package calls go through ids, the registry and the entry-point adapters.

## 5. Target layouts

```text
core/definitions/
  executable.py          # envelope and body models (T0)
  patch.py               # RFC 6902 subset, moved from synthesis
  schemas/executable.schema.json   # generated, package data
engine/executables/
  protocol.py registry.py resolver.py adapters.py
engine/kinds/call.py
agents/
  patterns/
    react.json reflexion.json plan_execute.json rewoo.json self_consistency.json debate.json
    orchestrator.json adapt.json evalopt.json maker_checker.json parallel_react.json
    native_fc.json prompt_chain.json meta_builder.json
  handlers/library.py    # names to handler factories
  pattern_agent.py       # PatternAgent
  fsm_definitions/       # kept: re-exports build_<x>_fsm
reasoning/
  strategies/
    analytical.json deductive.json inductive.json abductive.json analogical.json
    creative.json critical.json hybrid.json simple_calculator.json
    classifier.json orchestrator.json
workflows/
  loader.py callables.py steps/executable.py
synthesis/
  targets.py             # pattern and workflow compile targets
  policy.py              # plus from_json, to_json, the gate pipeline file
```

## 6. Phases

Every row ends in a test that fails on the parent commit. R1 comes first; R2 and R3 need R1; R4, R5 and R6 need R2; R7 needs roadmap 1.5.

### R0: guard rails, no behaviour change

| # | Change | Done when |
|---|---|---|
| R0.1 | Snapshot the canonical JSON and `fsm_id` of every built-in pattern (over a parameter matrix per builder) and every reasoning FSM | Fixtures committed; a test rebuilds and compares them |
| R0.2 | Inventory the decision anchors in every module this plan touches | Each anchor has a disposition: travels, superseded by a named new decision, or unaffected |
| R0.3 | List the `SRC_PINS` files of `scripts/synthesis_bench.py` this plan changes | The owner accepts that blocks s0 to s2 can no longer be re-reported (s2 already refuses); `scripts/bench_data/` is not edited |

### R1: the unified schema and validation (T0, T10)

| # | Change | Done when |
|---|---|---|
| R1.1 | Envelope and body models, schema version table, generated JSON Schema with its equality test | Every R0.1 definition wrapped in an envelope round-trips; a newer field in an older document warns or refuses like an FSM field |
| R1.2 | Patch applier moved to `core/definitions/patch.py`; synthesis imports it from there | Synthesis repair tests unchanged |
| R1.3 | The load sequence of Section 4.13; `fsm-llm-validate` reads executables | One test per step shows a bad document refused before any model call, with every failure of the file reported |

### R2: registry and resolver (T2, T9)

| # | Change | Done when |
|---|---|---|
| R2.1 | `Executable` protocol, FSM-kind adapters, `ExecutableRegistry` with sources, lookup, atomic reload, retention | A reload serves the new identity to new conversations while a running one finishes on the old; a failing file keeps its previous version |
| R2.2 | Aliases with weighted routes and the `executable/resolve` event | Assignment is deterministic per conversation id; the log names the identity served |
| R2.3 | The `fsm_llm.executable_kinds` entry-point group and its lazy loader | The import laziness test is unchanged; a kind with no installed adapter fails with a named error |

### R3: `call` and reasoning (T3 in FSMs, T4, T8 for frames)

| # | Change | Done when |
|---|---|---|
| R3.1 | `call` KindSpec, loader rule, G3, G4 and G6a hooks | A missing failure edge is refused; the flow analysis sees inputs and returns |
| R3.2 | Executor stack requests and the frame-mode `call` executor | A parent branches on `call_result.status`; the lock-order test is unchanged; a fork in the middle of a child resumes it |
| R3.3 | Reasoning strategies as files; orchestrator on `call`; `_StrategyStack` deleted | Every R0.1 reasoning fixture reproduced byte for byte; replies come in the same order; a strategy over budget ends `aborted` and the solve still answers |
| R3.4 | Selection over the registry; streaming `solve_problem` | A registered user strategy is chosen by the classifier and by `--type`; a stream yields strategy replies before the solve ends |
| R3.5 | `synthesize` on the shared executor body (roadmap P7.1) | The P7.1 criterion: the concierge example runs on a bare `API` |

### R4: agents from pattern files (T1)

| # | Change | Done when |
|---|---|---|
| R4.1 | Handler library and `PatternAgent` | A pattern file using only library handlers runs a tool loop end to end in a test |
| R4.2 | Built-in patterns as files with variants and one generator (`prompt_chain`); `build_<x>_fsm` become `instantiate` calls | Every R0.1 agent fixture reproduced byte for byte |
| R4.3 | `create_agent` and the registry; user directories; the HITL refusal | `create_agent("my_pattern")` works from a file in a user directory; a file that omits the approval gate for an approval-required tool is refused |
| R4.4 | Composition hooks; Verified and AutoMemory as presets | Their existing tests pass unchanged; the HITL suite passes unchanged |

### R5: workflows as executables (T3, T8 for delegated runs)

| # | Change | Done when |
|---|---|---|
| R5.1 | Workflow loader and callable registry | `load(serialize(w)) == w` for every step type with named callables |
| R5.2 | `ExecutableStep`, `target` on `AgentStep` and `ConversationStep`, the workflow adapter, delegated-mode `call` | A JSON workflow runs a strategy, a pattern and a library procedure and merges their outputs; an FSM `call` runs a workflow |
| R5.3 | Delegated-run handles and resume; workflow persistence (roadmap 1.8) | A killed process resumes a parent conversation and its child workflow from the log |

### R6: synthesis (T5, T6)

| # | Change | Done when |
|---|---|---|
| R6.1 | Policy and gate pipeline files | A file that drops or reorders a mandatory gate, or exceeds a ceiling, is refused; a host gate's failure enters the repair message |
| R6.2 | Library entries as `procedure` executables; the library as a registry source | A verified procedure is callable by id from a `call` state and a workflow step |
| R6.3 | `pattern:<id>` target | A compiled ReAct variant passes its gates and runs through `create_agent`; a generated handler binding is refused |
| R6.4 | `workflow` target with W1, W2, W5, W6 | A compiled workflow of executable steps passes; one naming a disallowed callable is refused |

### R7: async (T7), after roadmap 1.5

| # | Change | Done when |
|---|---|---|
| R7.1 | `arun` on every adapter | Each kind runs under one event loop with no worker threads |
| R7.2 | Parallel agents on asyncio tasks | No `ThreadPoolExecutor` in agents; cancellation reaches every branch |
| R7.3 | Concurrent G6b trials under admission control | Trials overlap and respect the admission limit |

## 7. Constraints and how they are met

| Constraint | How |
|---|---|
| Layer contracts (`.importlinter`) | No new layer; subpackages meet through ids and entry points, never imports; the named debts are retired by their roadmap rows (3.2, 6.6, P7.6), not by this plan |
| `import fsm_llm` laziness | Core and engine parts load with core; adapters load on first use through the entry point |
| No shims | `module_path_map.json` records every move; example-visible paths stay re-export packages |
| `examples/` untouched | Public names and `build_<x>_fsm` keep working; the examples tests are the check |
| Public API snapshot | Regenerated and reviewed in the commit that changes it |
| Wire strings | `synthesis_result`, `StopReason` values, context keys, `ReasoningType` values and `fsm_id` stay; `call_result` and `executable/resolve` are new |
| Decision anchors | Travel with their code; a retired rule (for example D-053, the step arithmetic of `_StrategyStack`) is superseded by a new decision in the same commit; `plans/ANCHORS.md` stays append-only |
| Bench pins | Accepted void per R0.3; the next registered block (roadmap P7.9) pins the new layout |
| Module-size cap | Every new module stays under it |

## 8. Roadmap changes this plan needs

These touch the roadmap, which is the owner's document; this plan does not edit it.

| Roadmap item | Change |
|---|---|
| C1 | Reworded: `FSMDefinition` stays the one FSM program; the executable envelope becomes the one file format for everything that runs, including workflows |
| C10 and row 5.5 | One entry-point group, `fsm_llm.executable_kinds`, ships before the rest of 5.5; pattern files reference handlers by name now (in the pattern body, not in `FSMDefinition`) |
| Row 1.8 | The declarative workflow subset is the `workflow` kind of the envelope |
| P7.1 | Implemented as `call` plus `synthesize` on one executor body |
| P7.2 | The library store also serves the registry |

## 9. Not done, and why

| Item | Reason |
|---|---|
| Python inside JSON (inline code, expressions beyond JsonLogic) | Rule 4; also G5's "no code in a definition" |
| One graph mixing FSM states and workflow steps | Calls across kinds give the same composition without a new execution model |
| Gate code as JSON | Gates are code; their order, parameters and host gates are data (Section 4.10) |
| A model choosing handler bindings | Rule 6 |
| Files dropped into the installed package | User directories instead (Section 4.3) |

## 10. Risks

| Risk | Mitigation |
|---|---|
| Stack requests from an executor (R3.2) break the lock order, rollback or the log fold | The stack module applies the requests; the lock-order, fork, resume and fold tests gate R3.2 |
| A converted definition changes a prompt by one character | Byte equality with the R0.1 fixtures before any builder is deleted |
| Pattern files weaken HITL | The grant check stays in the tool executor; a missing approval gate is refused; the HITL suite runs unchanged |
| Reload breaks a running or resumed conversation | Identity-keyed cache and retention; resume checks the identity |
| A user strategy or pattern returns keys the caller does not expect | The contract is checked against the flow analysis at load; `returns` copies only declared keys |
| Compiled workflows cannot be simulated | They top out at `draft` without human approval until the simulator drives workflows |
| Scope: this is several releases of work | Each phase ships alone; R0 to R2 deliver T0, T2, T9 and T10 without touching a subpackage's behaviour |

## Decisions for the owner

1. The C1 rewording in Section 8.
2. Name of the FSM kind: `call` (proposed), `invoke` or `subroutine`.
3. Whether strategy-returning functions may return `str` for user strategies (a `**BREAKING:**` entry if their annotated return type changes).
4. The bench pin consequence of R0.3.
5. Whether the `workflow` compile target (R6.4) ships before the simulator can drive workflows.

## Next steps

Start with R0.1 and R1. They change no runtime behaviour, they deliver the unified schema and up-front validation, and every later phase builds on them.
