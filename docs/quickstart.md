# Quick Start Tutorial

> Covers FSM-LLM v0.12.0

Welcome to FSM-LLM! This tutorial will have you building stateful conversations with LLMs in minutes.

## Prerequisites

- Python 3.11 or higher
- An LLM: a local Ollama model (name it with `LLM_MODEL=ollama_chat/<tag>`; there is no built-in default model), or any LiteLLM provider such as OpenAI with its API key (set `LLM_MODEL` and the provider's key)

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install llm-fsm

# With extensions
pip install "llm-fsm[monitor]"      # Real-time monitoring dashboard (fastapi, uvicorn, jinja2)
pip install "llm-fsm[all]"          # Every optional dependency
```

The distribution is `llm-fsm`; the import package is `fsm_llm`. `fsm-llm` on PyPI is a different, unrelated project: if you installed it by mistake, run `pip uninstall fsm-llm` first. The reasoning, workflows, agents, eval and synthesis extras add no packages (that code ships in every install); extras only add third-party dependencies.

Classification is built into the core package -- no extra install needed.

## 2. Set Your API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
# Or create a .env file and load it on purpose (nothing reads it unasked):
echo "OPENAI_API_KEY=your-api-key-here" > .env
# Python: fsm_llm.load_env()      CLI: fsm-llm --env-file .env --fsm your_fsm.json
```

Environment variables used by the library and the `fsm-llm` command:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_MODEL` | none | litellm model id. **Required by the `fsm-llm` CLI**; the `API` class uses it when no `model=` is passed, and raises `ModelNotConfiguredError` when neither is set |
| `LLM_TEMPERATURE` | `0.5` | CLI only |
| `LLM_MAX_TOKENS` | `1000` | CLI only |
| `FSM_PATH` | none | not usable with `fsm-llm`: `--fsm` is checked first and is mandatory |
| `OPENAI_API_KEY` etc. | none | provider keys, read by litellm |

Importing `fsm_llm` loads no `.env` file. Call `fsm_llm.load_env()` (it searches from the working directory upward, or takes a path, and does not replace a variable the process already has, except a blank one, which counts as unset; to keep a variable empty on purpose, do not name it in the file) or pass `--env-file PATH` to `fsm-llm`, `fsm-llm-eval`, `fsm-llm-meta`, `python -m fsm_llm.reasoning` or `fsm-llm-monitor`; otherwise `export` the variables.

## 3. Your First Bot

Create `hello_bot.py`:

```python
from fsm_llm import API

greeting_fsm = {
    "name": "friendly_greeter",
    "description": "Greets the user, learns their name, then says goodbye",
    "initial_state": "welcome",
    "states": {
        "welcome": {
            "id": "welcome",
            "description": "Greet the user and collect their name",
            "purpose": "Greet the user warmly and ask for their name",
            "extraction_instructions": "Extract the user's name.",
            "response_instructions": "Warmly greet the user and ask for their name.",
            "required_context_keys": ["name"],
            "transitions": [{
                "target_state": "personalized",
                "description": "Once the user has given their name",
                "conditions": [{
                    "description": "The name has been extracted",
                    "requires_context_keys": ["name"],
                    "logic": {"has_context": "name"}
                }]
            }]
        },
        "personalized": {
            "id": "personalized",
            "description": "Talk to the user by name",
            "purpose": "Give a personalized response using their name",
            "extraction_instructions": "Extract how the user says they are doing, as mood_reply.",
            "response_instructions": "Use their name and ask how they're doing.",
            "transitions": [{
                "target_state": "farewell",
                "description": "After user responds about their day"
            }]
        },
        "farewell": {
            "id": "farewell",
            "description": "Say goodbye",
            "purpose": "Wish them well and say goodbye",
            "response_instructions": "Wish them well using their name and say goodbye.",
            "transitions": []
        }
    }
}

api = API.from_definition(greeting_fsm, model="gpt-4o-mini")
conv_id, response = api.start_conversation()
print(f"Bot: {response}")

while not api.has_conversation_ended(conv_id):
    user_input = input("You: ")
    response = api.converse(user_input, conv_id)
    print(f"Bot: {response}")

print(f"\nCollected data: {api.get_data(conv_id)}")
```

Run it: `python hello_bot.py`

> **Gating**: `required_context_keys` only tells the pipeline which keys to extract in a
> state; it never blocks a transition. The `welcome -> personalized` move is held back by
> the transition **condition** (`requires_context_keys` plus a `has_context` rule), so the
> bot keeps asking until a name has been extracted. A transition with no conditions fires
> on the first message.

## 4. Add a Handler

Handlers add custom logic at 8 lifecycle points:

```python
from fsm_llm import API, HandlerTiming

api = API.from_definition(greeting_fsm, model="gpt-4o-mini")

def detect_mood(context):
    # Handlers see the context DATA (extracted keys plus a few "_" keys such as
    # _conversation_id), not the raw user message: read an extracted key.
    response = str(context.get("mood_reply", "")).lower()
    positive = ["good", "great", "wonderful", "amazing"]
    negative = ["bad", "terrible", "awful", "stressed"]
    if any(w in response for w in positive):
        return {"mood": "positive"}
    elif any(w in response for w in negative):
        return {"mood": "negative", "offer_help": True}
    return {"mood": "neutral"}

api.register_handler(
    api.create_handler("MoodDetector")
        .at(HandlerTiming.CONTEXT_UPDATE)
        .on_state("personalized")
        .do(detect_mood)
)
```

> **Timing note**: use `CONTEXT_UPDATE` to react to what was extracted *in* a state -- it
> fires after that state's extraction with `current_state` still set to it. `POST_PROCESSING`
> runs after the transition, so `.on_state(X)` there matches the **destination** state, not the
> state that did the extracting.

## 5. Streaming and Persistence

Stream the response tokens as they are generated, instead of waiting for the full reply:

```python
for chunk in api.converse_stream(user_input, conv_id):
    print(chunk, end="", flush=True)
```

Persist conversations across process restarts by attaching a session store:

```python
from fsm_llm import API, FileSessionStore

store = FileSessionStore("./sessions")
api = API.from_definition(greeting_fsm, model="gpt-4o-mini", session_store=store)
conv_id, response = api.start_conversation()
# State auto-saves after each converse() call. To resume later (e.g. a new process):
new_conv_id, state = api.restore_session(conv_id)
```

## 6. More Ways to Build

The conversation above is one way in. The rest of the library is built from the same parts, and the sections below show a builder, an agent, a workflow and the reasoning engine in a few lines each. `greeter.json` in the builder example stands for any FSM definition file, for example your own saved copy of the `friendly_greeter` dictionary above or the `greeter.json` of the project README.

### Assemble It Step by Step

Anything with more than a couple of settings can be put together by chaining calls. While you chain, nothing is checked; the final `.build()` checks everything at once and hands back a private copy, so one recipe can make many independent objects.

A conversation with a hook and a model:

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

An agent with a tool, an instruction and an iteration limit (a real run against the local model; the wording differs each time):

```python
from fsm_llm.agents import ConfiguredAgentBuilder, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


agent = (
    ConfiguredAgentBuilder()
    .set_pattern("react")
    .add_tool(add)
    .set_system_prompt("Answer in one short sentence.")
    .set_max_iterations(6)
    .build()
)
result = agent("What is 19 + 23? Use the add tool.")
print(result.answer, result.success, result.stop_reason)
# 19 + 23 equals 42, as calculated by the add tool. True answered
```

Mistakes come back together, in one error, instead of one at a time:

```python
from fsm_llm import APIBuilder, BuildError

try:
    APIBuilder().set_llm_option("model", "a").set_llm_option("max_history_size", 3).build()
except BuildError as err:
    print(*err.errors, sep="\n")
# option 'model' names a constructor parameter; use set_model()
# option 'max_history_size' names a constructor parameter; use set_max_history_size()
```

What you get from this style:

- Setters only record, so you can chain them in any order and pass the half-built recipe around.
- One `.build()` does all the checking and raises one error type. `BuildError` is a `ValueError` that lists every problem in `.errors` and keeps the underlying cause attached.
- The result is a private copy: change the recipe afterwards, or build again, and nothing already built moves. Objects you hand in (a model interface, a tool, a store) stay shared, not copied.
- Settings that pass straight through to the model library are accepted without a whitelist, and a name that clashes with a typed setter is refused with a message that names the setter to use.
- It is an alternative, not a replacement: the plain constructors and `from_file` still work everywhere.

The same shape builds conversations, agents of any pattern, workflows, agent graphs and handlers. The rules are in [API Reference](./api_reference.md).

### Create an Agent in One Line

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

### Run a Workflow (No LLM Needed)

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

### Solve a Problem with the Reasoning Engine

```python
from fsm_llm.reasoning import ReasoningEngine

engine = ReasoningEngine(model="ollama_chat/qwen3.5:4b")
solution, trace = engine.solve_problem("If all cats are mammals and Tom is a cat, what is Tom?")
print(solution)
print(trace["summary"])
```

## 7. Next Steps

### Try These Examples

```bash
python examples/basic/form_filling/run.py        # Collect structured info
python examples/basic/story_time/run.py           # Interactive storytelling
python examples/intermediate/book_recommendation/run.py  # Multi-branch conversations
```

### Key Concepts

1. **States** -- Each conversation step with a single purpose
2. **Transitions** -- Rules for moving between states (JsonLogic conditions)
3. **Context** -- Data collected and remembered across turns
4. **Handlers** -- Custom logic at 8 lifecycle points
5. **FSM Stacking** -- Nested sub-conversations via `push_fsm`/`pop_fsm`

### Useful Commands

```bash
fsm-llm --fsm your_fsm.json           # Run any FSM interactively (needs LLM_MODEL, --model or --env-file)
fsm-llm-validate --fsm your_fsm.json  # Validate FSM structure
fsm-llm-visualize --fsm your_fsm.json # ASCII visualization
fsm-llm-monitor                       # Launch monitoring dashboard
```

### Common Patterns

**Information gathering** -- Use `required_context_keys` to have the data extracted, and a transition condition (`requires_context_keys` + `has_context`) to stay in the state until it is collected.

**Branching logic** -- Multiple transitions with conditions route to different states.

**Error handling** -- Register a handler at `HandlerTiming.ERROR` for graceful recovery.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No API key found" | `echo $OPENAI_API_KEY` -- ensure it's set |
| "State not found" | Run `fsm-llm-validate --fsm your_fsm.json` |
| "Context key missing" | Add a debug handler at `POST_PROCESSING` to print context keys |
