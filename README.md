<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  </a>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue">
  <img alt="Version" src="https://img.shields.io/badge/version-0.13.0-green">
  <a href="https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml">
    <img alt="Tests" src="https://github.com/NikolasMarkou/fsm_llm/actions/workflows/python-package.yml/badge.svg">
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/NikolasMarkou/fsm_llm/main/images/fsm-llm-logo-1.png" alt="FSM-LLM Logo" width="500"/>
</p>

A language model call starts from nothing. A multi-turn conversation that must collect data, follow a set flow and decide the same way every time needs structure around the model. **FSM-LLM** is a Python framework for chatbots and agents that follow a fixed structure: you describe the conversation as a finite state machine in JSON, and the language model does the language work inside each state.

The split is simple. The **model** understands the user, pulls out facts and writes replies. The **state machine** decides where the conversation goes next, by rules evaluated in plain Python, so the same facts always lead to the same state. The **framework** keeps everything else: what was collected, the history, your hooks, sessions, budgets and an optional replayable log.

It is one package, `fsm_llm`, with a small core and subpackages on top: reasoning strategies, a workflow engine, agent patterns, a monitoring dashboard, an evaluation harness, runtime FSM synthesis and public test doubles. Every subpackage is built from the same parts as the core (stacked FSMs, message-free steps, one LLM layer), so there is one way to run a model, not one per subpackage.

<p align="center">
  <img src="https://raw.githubusercontent.com/NikolasMarkou/fsm_llm/main/images/fsm-diagram-example.png" alt="A form-collection FSM drawn by fsm-llm-visualize: welcome, name, email, age, a summary, a correction loop back to any field, then confirm and end" width="760"/>
</p>
<p align="center"><sub>A data-collection conversation with a correction loop. The states, the guards on each edge and the loop back are data in a JSON file, not prompt text.</sub></p>

---

## What You Can Build

| You want | What FSM-LLM gives you | Look at |
|----------|------------------------|---------|
| A form, intake or onboarding flow that never skips a required field | Per-state extraction, gates on the context (`requires_context_keys` plus a JsonLogic rule), correction handling | [form_filling](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/basic/form_filling) |
| Triage that routes by intent, with a deterministic fallback | `classification_extractions`, `Classifier`, `IntentRouter`; the model only breaks a tie | [smart_helpdesk](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/classification/smart_helpdesk) |
| A conversation that hands part of the job to a specialist and takes the result back | FSM stacking: `push_fsm` and `pop_fsm`, with only the keys you declare coming back | [e_commerce](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/advanced/e_commerce), [multi_level_stack](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/advanced/multi_level_stack) |
| Business logic around a chat: audit, analytics, lookups, side effects | Handlers at eight lifecycle points, with `continue` or `raise` error modes | [handler_hooks](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/advanced/handler_hooks) |
| A tool-using agent where a person approves risky calls | ReAct, ReWOO, Reflexion and more; human approval as a driver-only grant bound to one exact call | [hitl_approval](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents/hitl_approval), [react_search](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents/react_search) |
| A tree of agents with a manager and workers | `orchestrator`, `AgentGraph`, agents as tools of other agents | [hierarchical_orchestrator](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/agents/hierarchical_orchestrator) |
| A tutor that reasons in the middle of a chat | A reasoning engine that picks a strategy, runs it as stacked FSMs and validates the answer | [math_tutor](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/reasoning/math_tutor) |
| Long-running processes with events, timers and branches | An async workflow engine with typed steps, a Python DSL, waits and parallel steps, no model needed | [order_processing](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/workflows/order_processing) |
| A bot you can crash, replay and fork | An opt-in write-ahead event log, `API.fork` and `API.resume` (on main, for the next release) | [Durable runs](#durable-runs-log-fork-resume) |
| A bot you can unit test with no network | `fsm_llm.testing`: scripted models, a scripted user and state, path and context assertions | [Quick Start](#quick-start) |
| An FSM, workflow or agent you describe in chat | The meta-builder (`fsm-llm-meta`) | [build_fsm](https://github.com/NikolasMarkou/fsm_llm/tree/main/examples/meta/build_fsm) |

---

## Table of Contents
1.  [**Key Features**](#key-features): What it can do, with code.
2.  [**Why FSM-LLM?**](#why-fsm-llm): Where it fits and where it does not.
3.  [**Installation**](#installation): `pip install llm-fsm`, extras, picking a model.
4.  [**Quick Start**](#quick-start): A conversation, a fake model, a builder, an agent, a workflow.
5.  [**How It Works**](#how-it-works): The two-pass turn, the state kinds, the layers.
6.  [**In-Depth Documentation**](#in-depth-documentation): The docs site, the guides and per-package READMEs.
7.  [**Examples and Command-Line Tools**](#examples-and-command-line-tools): The runnable examples and the CLIs.
8.  [**Quality, CI and Release**](#quality-ci-and-release): Offline tests, import contracts, supply-chain checks.
9.  [**Project Structure**](#project-structure): A short tour of the repository.
10. [**Contributing**](#contributing): Where the rules live.
11. [**Status**](#status): Maturity and things to know.
12. [**License**](#license)
13. [**Acknowledgments**](#acknowledgments)
14. [**References**](#references): The work this library builds on.

---

## Key Features

### Flow is code, language is the model

Every transition is a rule over the collected context, written in JsonLogic and evaluated in plain Python. The operators cover comparison, boolean logic, arithmetic, `in`, `contains`, `if`, `var` with dot paths, and two additions for this library, `has_context` and `context_length`. The lowest `priority` among the passing transitions wins outright. Only a tie at the lowest priority asks the model, and then only to choose among the tied targets. If nothing passes, the conversation stays where it is.

```json
{
  "target_state": "adult_flow",
  "description": "The user is an adult",
  "priority": 10,
  "conditions": [
    {"description": "Age is known and at least 18",
     "requires_context_keys": ["age"],
     "logic": {">=": [{"var": "age"}, 18]}}
  ]
}
```

Numeric comparison coerces only plain decimal strings, so `"21"` passes `>= 18` and `"inf"` does not. A key you list in `handler_only_keys` cannot be written by the model, which keeps gates such as `is_admin` out of its reach.

### Extraction and classification, per state

Pass 1 of each turn extracts the fields the state declares: from `required_context_keys`, from the keys the outgoing edges need, or from explicit `field_extractions` with validation rules, retries and a confidence threshold. `classification_extractions` turn free text into one of your named intents. A wrong correction leaves the stored value in place and tells Pass 2 it was rejected. `context_scope` limits which keys a state's prompts can see.

### Conversations inside conversations

`push_fsm` starts a sub-conversation, `pop_fsm` ends it and merges back only the keys you declared. Stacks nest, each frame has its own history and turn budget, and an ended pushed FSM can be popped inside a message-free run so the parent continues on its own.

```python
api.register_handler(audit)                       # a handler from create_handler, see Quick Start
api.push_fsm(conv_id, "address_form.json", shared_context_keys=["zip"])
reply = api.converse("90210", conv_id)            # the child FSM collects the zip code
api.pop_fsm(conv_id)                              # back in the parent, with context["zip"]
```

### Hooks for everything around the model

Handlers run at `START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION` and `ERROR`. A fluent builder targets them by state, by target state, by context keys or by a condition. Return a dict and it merges into the context. A critical handler always raises; the others follow the `continue` or `raise` error mode, and `handler_timeout` bounds a slow one.

### Steps without a user message

`advance(conversation_id)` runs one turn with no incoming message and returns an `AdvanceResult`. `run_until_terminal(conversation_id, max_steps=..., max_seconds=...)` repeats it and raises `RunBudgetExceededError` when the budget is spent. Lifetime budgets on `API(max_steps=, max_seconds=)` are charged outside anything a model can write, so no extraction can reset them. Agents, workflow agent steps and the reasoning engine all run on these loops.

### One call primitive for tools and structured output

A state with the optional `completion` field makes its Pass 1 one `complete` call, with tool schemas or a JSON schema, over a transcript you own. The result lands in the context as `{kind, text, calls}` and your transitions read `<result_key>.kind`. Core runs no tool; your handlers or an agent loop do. Native function calling in the agents package is built on exactly this.

### Atomic turns and honest failures

A turn that fails is rolled back to its snapshot. A reply cut off at `max_tokens` is flagged `truncated`, logged once and never shown as an unreadable half envelope. Dead or contradictory state configuration is refused when the definition loads, and `FSMValidator` reports what the loader would accept but you probably did not mean. `fsm-llm-validate --upgrade` moves an older schema version (4.0 to 4.2) forward.

### Durable runs: log, fork, resume

On main, for the next release. Pass `event_log=` and every turn of a conversation tree is appended to one JSONL file, with large payloads in a content-addressed blob store, fsynced before each model call. A log that cannot be written refuses the call.

```python
from fsm_llm import API

api = API.from_file("greeter.json", event_log="logs")
conv_id, _ = api.start_conversation()
api.converse("Hi, I'm Alice.", conv_id)

branch_id, state = api.fork(conv_id, at_turn=0)   # a new conversation from the state before turn 1
api.converse("Hi, I'm Bob.", branch_id)           # a different answer; the original is untouched

# after a crash, in a new process, on the same log directory:
api = API.from_file("greeter.json", event_log="logs")
conv_id, state = api.resume(conv_id)              # same id; an in-flight tool is reported, never re-run
```

`fork` and `resume` rebuild the whole stack of pushed FSMs from the log. Privacy is a setting: `EventLog(root, privacy="values")` is the default, `"shape"` keeps ids, names and outcomes with no text (and then the log cannot be forked), and a `redact` hook handles free text. `EventLog.prune` is explicit retention and deletes nothing by default. Usage totals can be folded back out of the log with `fsm_llm.runtime.usage`.

### Check a definition before it runs

```python
from fsm_llm import FSMValidator, build_fsm_graph, to_mermaid
from fsm_llm.core.loaders import load_fsm_definition
from fsm_llm.core.verify.flow import analyze_data_flow, analyze_liveness

definition = load_fsm_definition("greeter.json")
print(to_mermaid(build_fsm_graph(definition)))        # a stateDiagram-v2 you can paste into a README
for finding in analyze_data_flow(definition) + analyze_liveness(definition):
    print(finding)                                    # for example shadowed-edge, permanent-tie, cycle-no-exit
```

The data-flow pass catches a state that reads a key nothing extracts. The liveness pass catches an edge a higher-priority edge always hides, a tie that can never resolve and a loop with no exit. `fsm-llm-visualize` draws any FSM as ASCII, Mermaid or Graphviz DOT.

### Agents, with a security boundary where it matters

`create_agent(pattern, tools, config=...)` builds an agent on the same FSM core. There is one registry of patterns, plus `AgentGraph` for a DAG of agents and a meta-builder that designs FSMs, workflows and agents by chat.

- **`fsm_llm.agents`**: the agent patterns (`react`, `rewoo`, `reflexion`, `plan_execute`, `prompt_chain`, `self_consistency`, `debate`, `orchestrator`, `adapt`, `evaluator_optimizer`, `maker_checker`, `reasoning_react`, `parallel_react`, `verified_react`, `auto_memory`, `native_fc`, `swarm`, `meta_builder`) plus `AgentGraph`.
- **Tools done properly**: the `@tool` decorator with annotations, enforced timeouts, tools loaded from folders of Python files (trusted sources only), standard operating procedures, retry and caching registries, and semantic tool selection.
- **MCP client**: `MCPToolProvider` discovers and registers tools from a stdio or HTTP MCP server (the `mcp` extra). There is no MCP server in the package yet.
- **Agent to agent over HTTP**: `AgentServer` exposes an agent at `/invoke`, `/stream`, `/health` and `/info` with an optional API key, and `RemoteAgentTool` calls it as a tool. It is a small HTTP protocol, not the full A2A specification.
- **Memory**: semantic memory with embeddings, working-memory tools, auto-memory agents and a read-only OKF backend.

Human approval is enforced by the driver, not the prompt:

```python
from fsm_llm.agents import AgentConfig, HumanInTheLoop, create_agent, tool


@tool(requires_approval=True)
def wire_money(amount: int) -> str:
    """Send money."""
    return f"sent {amount}"


hitl = HumanInTheLoop(approval_callback=lambda request: input("Allow this call? [y/N] ") == "y")
agent = create_agent("react", [wire_money], config=AgentConfig(model="openai/gpt-4o-mini"), hitl=hitl)
```

A gated tool runs only with a grant bound to that exact call. The model cannot set it, a caller's `initial_context` cannot set it, and one approval covers one call. `react`, `reflexion` and `reasoning_react` support it; the patterns that cannot enforce it refuse a registry that holds a gated tool instead of ignoring the flag.

### Reasoning, workflows and synthesis

- **`fsm_llm.reasoning`**: a solver that picks one of the reasoning strategies, runs it as stacked FSMs inside one message-free core run, validates the answer and retries up to 3 times. `ReasoningEngine.solve_problem(problem)` returns the solution and a trace.
  The nine strategies are `simple_calculator`, `analytical`, `deductive`, `inductive`, `abductive`, `analogical`, `creative`, `critical` and `hybrid`. A classifier FSM recommends one; `preferred_reasoning_type` or `--type` overrides it.
- **`fsm_llm.workflows`**: an async in-memory engine with typed steps (`auto_step`, `api_step`, `condition_step`, `switch_step`, `llm_step`, `parallel_step`, `wait_event_step`, `timer_step`, `retry_step`, `conversation_step`, `agent_step`) and a Python DSL. Waits match events by a correlation key and can time out into another step; a parallel step gives each child its own copy of the context. It runs without an LLM unless you add LLM steps.
- **`fsm_llm.synthesis`** (experimental): a host FSM names a goal in a `synthesize` state, a model writes a child FSM for it, and the child runs only after gates: it loads, passes the validator, a data-flow check, a liveness check, a policy allow-list, a dead-edge check through the real evaluator, an optional bounded simulation and an optional approval. A file-backed library stores it and a runner pushes it as a child of the host conversation. `verified` means those gates passed over a model-written spec, not that the procedure is correct. Bench blocks `s0`, `s1` and `s2` were run on one model and all three verdicts are INCONCLUSIVE. See the [synthesis guide](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/synthesis.md).

### Knowledge bundles and Agent Skills

- **Open Knowledge Format (OKF)**: `fsm_llm.okf` reads, searches, validates and writes OKF bundles (a directory of markdown files with YAML frontmatter). `fsm_llm.engine.okf_bridge.create_okf_handler()` puts capped concept summaries into a plain FSM's context, `fsm_llm.agents` adds read-only OKF tools and bundle exporters, and `fsm-llm-okf` runs it from the shell. It follows OKF v0.2 (pre-1.0 and mutable).
- **Agent Skills**: `fsm_llm.skills` reads folders with a `SKILL.md` from the paths you name (the open standard at agentskills.io, unversioned and mutable). `fsm_llm.agents.create_skill_tools` gives an agent `skill_list`, `skill_load` and `skill_read`, and `fsm-llm-skills` validates, lists and shows them.
- **Untrusted by design**: bundle and skill text reaches a model only as capped, control-stripped tool output or labelled summaries, never as a tool description, instruction or persona. Nothing in a bundle or skill folder is executed, and `allowed-tools` is reported, not enforced.

### Watch it, score it, test it

- **`fsm_llm.monitor`**: a FastAPI dashboard (REST, WebSocket, a vanilla-JS single-page app) with events, logs, metrics, a visualizer, a chat console and the meta-builder. `InstanceManager.attach_api(api)` lets it watch your own `API`. It launches example FSMs from a checkout, a set of agent patterns with stub tools and two demo workflows; it is a development tool. Optional OpenTelemetry export (`--otel`).
- **`fsm_llm.eval`**: scores the examples 0 to 4, or plays scripted conversations N times and reports pass rates with Wilson confidence intervals (`run_dataset`, `run_cases`, `wilson_ci`).
- **`fsm_llm.testing`**: public test doubles for your own code, and it never imports pytest. `ScriptedLLM`, `MockLLM2Interface` and `PromptGroundedLLM` are fake `LLMInterface` implementations that also answer the async twins; reply builders such as `final_reply` and `tool_call_reply`; a `ScriptedUser` that drives a conversation; and the assertions `assert_state`, `assert_context`, `assert_path` and `assert_result`.

### One LLM layer, safe defaults

- **One LLM layer**: litellm is imported in exactly one module. `LLMInterface.complete(CompletionRequest)` is the single primitive for tool calling, plain and structured completion, with per-interface call and token counters (`usage()`) and an embedder (`LiteLLMEmbedder`). Any litellm model string works.
- **Async at the model layer**: every `LLMInterface` method has an `a*` twin (`acomplete`, `agenerate_response`, `aextract_field` and so on), as do `LiteLLMEmbedder.aembed` and `Classifier.aclassify`. `LiteLLMInterface` is natively async, the sync names are thin wrappers over it, and a sync caller needs no event loop. `API` has async turns too (`astart_conversation`, `aconverse`, `aadvance`, the two streams and `arun_until_terminal`, with the sync names as thin wrappers over the same bodies); a turn cancelled mid-flight is rolled back whole, and a second turn on a busy conversation is refused, never queued. Handlers and tools still run inline on the loop. On main, for the next release.
- **No hidden configuration**: the library has no built-in default model (`model=`, then `LLM_MODEL`, else `ModelNotConfiguredError`; no CLI, no monitor route and no example script ends on a default model), importing `fsm_llm` loads no `.env` file (you call `load_env()` or pass `--env-file`), and the library logs nothing until you ask.
- **Context security**: internal keys (prefixes `_`, `system_`, `internal_`, `__`) and secret-looking entries are kept out of prompts; restores seed only public keys, so a tampered session file cannot forge an approval grant; file paths from bundles and skills are confined to their root. The compromised litellm 1.82.7 and 1.82.8 are below the required version, and `make audit` scans for malicious `.pth` files.
- **Enforced layering**: a layer package imports only the layers below it, written as import contracts (`.importlinter`) that a test and CI run, so the architecture cannot drift unnoticed.

---

## Why FSM-LLM?

A plain loop of "send history, get reply" is the right tool for open-ended chat. It gets harder to trust once the conversation has to do something specific:

| Concern | A prompt-only loop | FSM-LLM |
|---------|--------------------|---------|
| "What happens next?" | Whatever the model decides this time | A rule you can read, test and diagram; the model breaks only an exact tie |
| Required data | Hope the prompt makes the model ask | An edge guarded by a gate does not fire until the keys are present and the rule passes |
| A failed model call | Half-applied state | The turn rolls back to its snapshot |
| Risky actions | A sentence in the prompt | A grant the model cannot set, bound to one call |
| Reproducing a bug | Re-run and hope | Fork the log at the turn that went wrong (on main) |
| Testing | Needs a live model | A scripted model, a scripted user and path assertions |
| Changing the flow | Edit and re-test a long prompt | Edit a JSON definition, validate it, diagram it |

FSM-LLM splits the work three ways:

*   **The LLM handles language**: understanding the user, pulling out facts, writing replies.
*   **The state machine handles flow**: which state you are in and where you go next is a rule, not a mood. The LLM breaks a tie between two equal rules and nothing else.
*   **The framework handles state**: what has been collected, history, your own hooks, and saving and restoring sessions.

Use it when the conversation has steps: forms and intake, onboarding, support triage, tutoring flows, anything where "what must happen next" is a rule.

Do not use it for open-ended chat with no structure (a plain LLM call is simpler), or when you need a hosted, production-hardened agent platform. The project is young, and one subpackage, synthesis, is explicitly experimental (see [Status](#status)).

---

## Installation

> **Note:** This library requires Python 3.11 or newer (tested on 3.11 to 3.15). It is published on PyPI as `llm-fsm`; the import package is `fsm_llm`. `fsm-llm` on PyPI is a different, unrelated project: if you installed it by mistake, run `pip uninstall fsm-llm` first.

1.  **Create an environment and install:**
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install llm-fsm
    ```

2.  **Optional extras:**
    `import fsm_llm` loads only the core, and every subpackage ships in every install. An extra only adds third-party dependencies.

    | Extra | Adds |
    |-------|------|
    | `reasoning`, `agents`, `workflows`, `eval`, `synthesis`, `testing` | No third-party packages (the code ships in every install) |
    | `monitor` | fastapi, uvicorn, jinja2 |
    | `mcp` | mcp |
    | `otel` | opentelemetry-api, opentelemetry-sdk |
    | `a2a` | httpx (`AgentServer` also needs fastapi, which the `monitor` extra installs) |
    | `all` | Every extra above |
    | `dev` | The test, lint and type tools (for contributors, see [CONTRIBUTING](https://github.com/NikolasMarkou/fsm_llm/blob/main/CONTRIBUTING.md)) |

    ```bash
    pip install "llm-fsm[monitor]"
    pip install "llm-fsm[all]"
    ```

    The core install needs loguru, litellm, pydantic, python-dotenv and pyyaml (which parses OKF and Agent Skills frontmatter).

3.  **Pick a model:**
    ```bash
    export LLM_MODEL=ollama_chat/qwen3.5:4b      # or openai/gpt-4o-mini with OPENAI_API_KEY set
    ```
    The model is the `model=` argument, then `LLM_MODEL`; there is no built-in default, and with neither set the call raises `ModelNotConfiguredError` (a CLI prints its message and exits 1). No `.env` file is read unless you ask: call `fsm_llm.load_env()` (or `fsm_llm.load_env("path/to/.env")`) in Python, or pass `--env-file PATH` to a CLI that needs a model (`fsm-llm`, `fsm-llm-eval`, `fsm-llm-meta`, `python -m fsm_llm.reasoning`, `fsm-llm-monitor`); a variable the process already has is not replaced unless `override=True`, except an empty or whitespace-only one, which counts as unset and is filled when the file gives it a value.

4.  **Verify the installation:**
    ```bash
    python -c "import fsm_llm; print('Installation successful!')"
    ```

---

## Quick Start

### 1. Define and Run a Conversation

Define the conversation (`greeter.json`). It asks for a name, then moves to a final state:

```json
{
  "name": "Greeter",
  "description": "Learns the user's name, then says goodbye",
  "initial_state": "ask_name",
  "persona": "A friendly assistant",
  "states": {
    "ask_name": {
      "id": "ask_name",
      "description": "Find out the user's name",
      "purpose": "Learn the user's name",
      "extraction_instructions": "Extract the user's name if they gave it",
      "response_instructions": "Greet the user and ask for their name if you do not know it yet",
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
      "response_instructions": "Say a short goodbye to the user by name. Do not ask a question"
    }
  }
}
```

Run it:

```python
from fsm_llm import API

api = API.from_file("greeter.json")   # no model given: env LLM_MODEL, else ModelNotConfiguredError
conv_id, opening = api.start_conversation()
print("bot:", opening)
print("bot:", api.converse("Hi, I'm Alice.", conv_id))
print(api.get_data(conv_id))
print(api.get_current_state(conv_id), api.has_conversation_ended(conv_id))
```

A real run against a local Ollama `qwen3.5:4b` (the wording differs on every run):

```text
bot: Hello! How do you do? I'm here to get to know you better. What's your name?
bot: Nice meeting you, Alice! It was a pleasure getting to know you.
{'name': 'Alice'}
goodbye True
```

To use a hosted model, pass `model=` (for example `API.from_file("greeter.json", model="openai/gpt-4o-mini")`) and set the provider key (`OPENAI_API_KEY`).

A state with no transitions (`goodbye`) is terminal. `required_context_keys` only tells Pass 1 what to extract; it never blocks a transition, so the transition above carries its own condition (`requires_context_keys` plus `logic`).

### 2. Try It Without a Model

Pass your own `llm_interface` and nothing touches the network. `fsm_llm.testing` ships a scripted double, the one the repository's own tests use:

```python
from fsm_llm import API
from fsm_llm.testing import ScriptedLLM

llm = ScriptedLLM(
    extraction_data={"name": "Alice"},
    responses={"ask_name": "What is your name?", "goodbye": "Goodbye, Alice!"},
)
api = API.from_file("greeter.json", llm_interface=llm)
conv_id, opening = api.start_conversation()
print(opening)                                                # What is your name?
print(api.converse("I am Alice", conv_id))                    # Goodbye, Alice!
print(api.get_data(conv_id), api.get_current_state(conv_id))  # {'name': 'Alice'} goodbye
```

`API` refuses `model=`, `temperature=` and similar settings next to `llm_interface=`, because the interface owns them.

A scripted user and the assertions turn the same bot into a test in a few lines:

```python
from fsm_llm import API
from fsm_llm.testing import (
    ScriptedLLM, ScriptedUser, assert_context, assert_path, assert_state,
)

llm = ScriptedLLM(extraction_data={"name": "Alice"}, responses={"goodbye": "Bye!"},
                  default_response="Hi!")
run = ScriptedUser(["I am Alice"]).run(API.from_file("greeter.json", llm_interface=llm))
assert_state(run, "goodbye")
assert_context(run, name="Alice")
assert_path(run, ["ask_name", "goodbye"])
print("the bot passes", run.states)                           # the bot passes ('ask_name', 'goodbye')
```

### 3. Assemble It Step by Step

Anything with more than a couple of settings can be put together by chaining calls. While you chain, nothing is checked; the final `.build()` checks everything at once and hands back a private copy, so one recipe can make many independent objects.

```python
from fsm_llm import APIBuilder, HandlerTiming, create_handler

audit = (
    create_handler("audit")
    .at(HandlerTiming.POST_TRANSITION)
    .do(lambda context: {"audited": True})
)

api = (
    APIBuilder()
    .set_definition("greeter.json")
    .set_model("ollama_chat/qwen3.5:4b")
    .set_temperature(0.2)
    .add_handler(audit)
    .build()
)
```

Mistakes come back together: `build()` raises one `BuildError` (a `ValueError`) that lists every problem in `.errors`. The result is a private copy, and the plain constructors and `from_file` still work everywhere. The same shape builds conversations, agents of any pattern, workflows, agent graphs and handlers (see the [API reference](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/api_reference.md)).

### 4. Create an Agent in One Line

```python
from fsm_llm.agents import AgentConfig, create_agent, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


agent = create_agent("react", [add], config=AgentConfig(model="ollama_chat/qwen3.5:4b"))   # or set LLM_MODEL
result = agent("What is 19 + 23? Use the add tool.")
print(result.answer, result.success, result.stop_reason)   # 19 + 23 equals 42. True answered
```

`success` is `True` only when the run reached its goal. A forced stop still returns its last answer, with `success=False` and a `stop_reason`.

### 5. Run a Workflow (No LLM Needed)

This prints `WorkflowStatus.COMPLETED`:

```python
import asyncio
from fsm_llm.workflows import WorkflowEngine, auto_step, condition_step, create_workflow_builder

wf = (
    create_workflow_builder("orders", "Order check")
    .set_initial_step(auto_step("load", "Load order", next_step_id="route",
                                action=lambda ctx: {"amount": 1500}))
    .add_step(condition_step("route", "Big order?", condition=lambda ctx: ctx["amount"] >= 1000,
                             true_step_id="review", false_step_id="done"))
    .add_step(auto_step("review", "Manual review", next_step_id="done"))
    .add_step(auto_step("done", "Finish", next_step_id=""))
    .build()
)


async def main():
    engine = WorkflowEngine()
    engine.register_workflow(wf)
    instance_id = await engine.start("orders")
    print(engine.get_status(instance_id))


asyncio.run(main())
```

### 6. Solve a Problem with the Reasoning Engine

```python
from fsm_llm.reasoning import ReasoningEngine

engine = ReasoningEngine(model="ollama_chat/qwen3.5:4b")
solution, trace = engine.solve_problem("If all cats are mammals and Tom is a cat, what is Tom?")
print(solution)
print(trace["summary"])
```

More (handlers, streaming, sessions, every builder) is in the [Quick Start guide](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/quickstart.md).

---

## How It Works

Every user message goes through two LLM passes:

```text
  user message
       |
       v
  Pass 1: extract data (LLM) --> update context
       |
       v
  transition rules (JsonLogic, plain Python, no LLM)
       |
       +-- one rule wins ----------------> new state --+
       +-- several tie at the lowest ----> LLM picks one -> new state --+
       +-- none passes ------------------> stay in state ---------------+
                                                                        |
                                                                        v
                                         Pass 2: write the reply from the final state (LLM)
```

- The context is a dictionary of everything collected so far in one conversation.
- Transition rules use JsonLogic, for example `{">=": [{"var": "age"}, 18]}`. If several transitions pass, the lowest `priority` number wins outright. The LLM is asked only when two or more tie at that lowest priority. If none pass, the conversation stays put.
- Pass 2 runs after the transition, so the reply comes from the state the conversation is actually in. A state with empty `response_instructions` makes no LLM call, replies with `""` and adds nothing to the history.
- The turn is atomic. If a pass fails, the context, history and state go back to the snapshot taken at the start of the turn. A concurrent or re-entrant `converse` on the same conversation raises `FSMError` instead of interleaving.
- With an event log on, the exact request of every model call is written and fsynced before the call, so a run can be rebuilt, forked or resumed from the log.

### State kinds

| Kind | Pass 1 does | Use it for |
|------|-------------|------------|
| `conversation` | Extracts fields and classifications from the user message, then Pass 2 writes the reply | Forms, intake, support flows (the default) |
| `completion` | One `complete` call with tools or a JSON schema over a transcript you own; the result `{kind, text, calls}` lands in the context | Tool calling and structured turns; your handlers run the tools |
| `synthesize` | Core only loads and validates the field; `fsm_llm.synthesis.ProcedureRunner` has a model write a child FSM for the goal and runs it behind the gates | A sub-conversation that cannot be written ahead of time |

### The layers

The core is built in layers, and a layer imports only the rows below it (the contracts are in `.importlinter`). The subpackages sit on top and are imported by name, so `import fsm_llm` loads only the core:

```text
  monitor  eval  agents  synthesis  workflows  reasoning  testing    <- subpackages: from fsm_llm.agents import ...
  ---------------------------------------------------------------
  engine      API, FSMManager, the turn pipeline, state kinds, classifier, sessions
  handlers    HandlerSystem, create_handler
  policy      approval vocabulary, HumanInTheLoop
  tools       ToolRegistry, @tool, timeouts
  context  okf  skills      independent siblings
  runtime     event log, JSONL and blob stores, turn scope, usage fold (opt-in)
  llm         LLMInterface, LiteLLMInterface: the one litellm import
  core        definitions, JsonLogic expressions, validator, diagrams
  kernel      aio, constants, logging, security, text, env, frontmatter, memory
```

Maturity differs by subpackage: reasoning, agents and synthesis are tested offline against scripted models, so live behavior depends on the model; the `eval` scores are a heuristic that overstates quality; `monitor` needs the `monitor` extra. Each [per-package README](#in-depth-documentation) lists its known limits.

**Evaluation honesty.** The examples scorer reads each run's output with simple rules and gives 0 to 4. It overstates quality, and runs from before and after the 2026-09-29 restructure are not comparable. The method and the run log are in [EVALUATE.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/EVALUATE.md). Offline tests (the default) replace the model with a fake, so they say nothing about model quality.

---

## In-Depth Documentation

-   **Documentation site:** [https://nikolasmarkou.github.io/fsm_llm/](https://nikolasmarkou.github.io/fsm_llm/) (the guides below, built with MkDocs on every pull request and versioned per release).
-   **Start here, in this order:** [Quick Start](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/quickstart.md) (zero to a running conversation, then builders, an agent, a workflow and the reasoning engine), [Concepts](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/concepts.md) (what is in the box, where it fits), [FSM Design](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/fsm_design.md) (states, transitions, priorities, classification, `completion` and `synthesize`) and [Handlers](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/handlers.md) (the hook points).
-   **Going deeper:** [Architecture](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/architecture.md) (the pipeline, the layers, the step driver), [Testing](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/testing.md) (test your own bot offline), [Runtime FSM synthesis](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/synthesis.md), [API Reference](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/api_reference.md) and the [command-line tools](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/cli.md).
-   **Per-package READMEs:** [core](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/README.md), [reasoning](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/reasoning/README.md), [workflows](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/workflows/README.md), [agents](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/agents/README.md), [monitor](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/monitor/README.md), [eval](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/eval/README.md), [synthesis](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/synthesis/README.md) and [testing](https://github.com/NikolasMarkou/fsm_llm/blob/main/src/fsm_llm/testing/README.md).
-   **Project records:** [CHANGELOG](https://github.com/NikolasMarkou/fsm_llm/blob/main/CHANGELOG.md), [EVALUATE](https://github.com/NikolasMarkou/fsm_llm/blob/main/EVALUATE.md) (method and run log), [CONTRIBUTING](https://github.com/NikolasMarkou/fsm_llm/blob/main/CONTRIBUTING.md), [SECURITY](https://github.com/NikolasMarkou/fsm_llm/blob/main/SECURITY.md) and the [2026 roadmap](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/2026_roadmap.md).

---

## Examples and Command-Line Tools

### Examples

The examples are runnable and grouped by category. Each is a folder with `run.py` and its FSM JSON. They are in the repository, not in the installed package, so get them from a checkout of the source and run one:

```bash
python examples/basic/simple_greeting/run.py     # needs a model: an OpenAI key, or a local Ollama
```

| Category | Highlights |
|----------|------------|
| Basic | simple_greeting, form_filling, multi_turn_extraction |
| Intermediate | book_recommendation, product_recommendation, adaptive_quiz |
| Advanced | e_commerce (FSM stacking), support_pipeline, concurrent_conversations |
| Classification | intent_routing, smart_helpdesk, multi_intent |
| Reasoning | math_tutor |
| Workflows | order_processing, parallel_steps, conditional_branching |
| Agents | react_search, plan_execute, reflexion, debate, orchestrator |
| Meta | build_fsm, build_workflow, build_agent |

The examples are evaluation baselines: `fsm-llm-eval examples` runs them in parallel and scores each. The list with links is the [examples page](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/examples.md).

### Command-Line Tools

| Command | Description |
|---------|-------------|
| `fsm-llm --fsm <path.json> [--model M] [--env-file PATH]` | Chat with an FSM interactively (`--mode validate` and `--mode visualize` also exist) |
| `fsm-llm-visualize --fsm <path.json> [--format ascii\|mermaid\|dot]` | Draw an FSM |
| `fsm-llm-validate --fsm <path.json> [--upgrade]` | Check an FSM definition for problems; `--upgrade` rewrites an older schema version after validating |
| `fsm-llm-monitor` | Web dashboard at http://127.0.0.1:8420 (`--host`, `--port`, `--no-browser`, `--api-key`, `--otel`, `--env-file`) |
| `fsm-llm-meta [--model M] [--env-file PATH] [--output FILE]` | Build FSMs, workflows or agents by chatting |
| `fsm-llm-okf <validate\|list\|show\|search\|export>` | Check, browse, search and export Open Knowledge Format bundles |
| `fsm-llm-skills <validate\|list\|show> PATH...` | Check (`--strict` for API portability), list and read Agent Skills folders |
| `fsm-llm-eval examples [--category C] [--fail-under PCT] [--env-file PATH]` | Score every example 0 to 4 |
| `fsm-llm-eval run <dataset.json\|.jsonl> [--trials N] [--env-file PATH]` | Run scripted conversations and report pass rates |
| `python -m fsm_llm.reasoning "problem" [--type T] [--model M] [--env-file PATH]` | Solve a problem with the reasoning engine (no console script) |

The CLIs that call a model need one: `--model/-m NAME`, else `LLM_MODEL` in the environment, else the file given with `--env-file PATH` (the process environment wins over the file); with none they print one `Error:` line and exit 1. `fsm-llm-monitor` starts without a model, and its launch routes answer 400 until one is set. The commands that call no model (`fsm-llm-okf`, `fsm-llm-skills`, `fsm-llm-validate`, `fsm-llm-visualize`) have no `--env-file`. Exit codes: 0 ok, 1 failure, 130 Ctrl-C; `fsm-llm-eval` exits 2 below `--fail-under`. Every option is on the [command-line page](https://github.com/NikolasMarkou/fsm_llm/blob/main/docs/cli.md).

---

## Quality, CI and Release

- **Offline by default, enforced by contracts.** The default test run needs no network and no model (live tests skip without a local Ollama). Import contracts, a module-size guard, a public-API snapshot and a decision-anchor gate run as tests or CI steps.
- **A CI matrix.** `python-package.yml` runs `lint` (ruff, mypy, the `.pth` audit, the import contracts, the workflow linters), `supply-chain` (`pip-audit` over the installed environment, with a canary that must fail), `dependency-review` on pull requests, and `test` on Linux with Python 3.11 to 3.15 and on Windows with 3.11 and 3.15. Runners and actions are pinned, and a coverage ratchet guards the Linux 3.12 leg.
- **Pinned dependencies.** `constraints.txt` pins every direct dependency exactly; a weekly job proposes a newer litellm or mcp only when it is stable, at least 7 days old and green on the suite, and never merges by itself.
- **Releases come from a tag.** One pushed tag `vX.Y.Z` builds the package once, writes an SBOM, publishes to PyPI by trusted publishing with attestations and creates the GitHub release; nothing publishes from a developer machine. The same tag deploys the docs site.

---

## Project Structure

One distribution (`llm-fsm`, setuptools, src layout): the directory `src/fsm_llm/` is the import package `fsm_llm`. Beside it: `tests/` (one folder per subpackage), `examples/` (the runnable examples, the evaluation baselines), `docs/` (the pages of the site), `evaluation/` (datasets and recorded runs), `scripts/` (supply-chain audit, benches) and `.github/workflows/` (CI, docs, pin refresh, release). The full tree, one line per package, is in [CONTRIBUTING](https://github.com/NikolasMarkou/fsm_llm/blob/main/CONTRIBUTING.md).

---

## Contributing

Contributions are welcome, from a typo fix to a new agent pattern. Setup, the test, lint and type commands, the rules for new code, the release procedure and the open items worth picking up are in [CONTRIBUTING](https://github.com/NikolasMarkou/fsm_llm/blob/main/CONTRIBUTING.md). Report vulnerabilities as described in [SECURITY](https://github.com/NikolasMarkou/fsm_llm/blob/main/SECURITY.md).

---

## Status

- Version 0.13.0. The event log, `API.fork`, `API.resume`, turn budgets and the async twins described above are on main and ship in the next release; the badge and the PyPI package are 0.13.0.
- Maturity: the core engine, handlers, stacking, validation and the test doubles are the settled part. Agents, reasoning and workflows are tested offline against scripted models, so live behavior depends on the model. Synthesis is experimental. The monitor is a development tool. MCP support is a client only, and human approval is enforced for `react`, `reflexion` and `reasoning_react`.
- Most tests are offline and use a fake model, so they check logic and wiring, not model quality. Evaluation scores are a heuristic and overstate quality; the recorded runs in [EVALUATE.md](https://github.com/NikolasMarkou/fsm_llm/blob/main/EVALUATE.md) used a 4B local model.
- The library logs nothing until you call `setup_logging()` or `enable_debug_logging()`; both also replace loguru's default handler, so a library record prints once.
- One conversation handles one message at a time: a second `converse` on a conversation that is mid-turn raises `FSMError`.

---

## License

This project is licensed under the **Apache License 2.0**.

- **Permissive**: you may use, modify and distribute the library, including in commercial and closed-source products.
- **Attribution**: redistributions must retain the copyright notice and a copy of the license, and modified files must carry a notice stating that they were changed.
- **Patent grant**: contributors grant an express patent license covering their contributions.

See the [LICENSE](https://github.com/NikolasMarkou/fsm_llm/blob/main/LICENSE) file for complete details.

---

## Acknowledgments

FSM-LLM stands on open-source work: [litellm](https://github.com/BerriAI/litellm) for the provider layer, [Pydantic](https://docs.pydantic.dev) for the models, [loguru](https://github.com/Delgan/loguru) for logging, [JsonLogic](https://jsonlogic.com) for transition rules, and [Ollama](https://ollama.com) for the local models the evaluation runs used. Thanks to everyone who has reported issues and run the examples against other models.

---

## References

The agent patterns, reasoning strategies and standards in this library follow published work and open specifications. Each pattern's module documents how it differs from the original.

<details>
<summary><b>Agent Patterns, Reasoning and Evaluation</b></summary>

-   **ReAct: Synergizing Reasoning and Acting in Language Models**: Yao, S., et al. (2022). *ICLR 2023*.
-   **ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models**: Xu, B., et al. (2023).
-   **Reflexion: Language Agents with Verbal Reinforcement Learning**: Shinn, N., et al. (2023). *NeurIPS*.
-   **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models**: Wang, L., et al. (2023). *ACL*.
-   **Self-Consistency Improves Chain of Thought Reasoning in Language Models**: Wang, X., et al. (2022). *ICLR 2023*.
-   **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**: Wei, J., et al. (2022). *NeurIPS*.
-   **Improving Factuality and Reasoning in Language Models through Multiagent Debate**: Du, Y., et al. (2023).
-   **Building Effective Agents** (prompt chaining, orchestrator-workers, evaluator-optimizer): Anthropic (2024).
-   **Probable Inference, the Law of Succession, and Statistical Inference**: Wilson, E. B. (1927). *Journal of the American Statistical Association*. The source of the Wilson confidence interval used by `wilson_ci`.
</details>

<details>
<summary><b>Standards and Specifications</b></summary>

-   **Model Context Protocol (MCP)**: the protocol behind the `mcp` extra.
-   **Agent2Agent (A2A) Protocol**: the protocol behind the `a2a` extra.
-   **Open Knowledge Format (OKF) v0.2**: pre-1.0 and mutable.
-   **Agent Skills** (agentskills.io): the `SKILL.md` folder standard, unversioned and mutable.
-   **JsonLogic**: the rule format used for transition conditions.
-   **OpenTelemetry**: the telemetry format behind the `otel` extra.
</details>
