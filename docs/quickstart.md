# Quick Start

Welcome to FSM-LLM! This tutorial gets you to a running, cost-aware LLM program in under five minutes.

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key (or another provider supported by [litellm](https://docs.litellm.ai/docs/providers))

## 1. Installation

```bash
pip install fsm-llm

# Optional extras
pip install fsm-llm[monitor]    # Web dashboard
pip install fsm-llm[mcp]        # Model Context Protocol for agents
pip install fsm-llm[otel]       # OpenTelemetry exporter
pip install fsm-llm[all]        # Everything
```

`reasoning`, `agents`, and `workflows` are pure-Python stdlib subpackages — already included in the core install.

## 2. Set Your API Key

```bash
export OPENAI_API_KEY="sk-..."
# Or create a .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

## 3. The unified entry point — `Program`

Every program — chatbot, pipeline, long-context task — goes through one entry point:

```python
from fsm_llm import Program
```

There are three constructors. **Mode is fixed at construction**, and the verb is always `program.invoke(...)` returning a `Result`.

## 4. Your first chatbot — `Program.from_fsm`

Describe states and transitions in JSON, then load it.

```python
from fsm_llm import Program

greeting_fsm = {
    "name": "friendly_greeter",
    "initial_state": "welcome",
    "states": {
        "welcome": {
            "id": "welcome",
            "purpose": "Greet the user warmly and ask for their name",
            "response_instructions": "Warmly greet the user and ask for their name.",
            "transitions": [{
                "target_state": "personalized",
                "description": "After user provides their name"
            }]
        },
        "personalized": {
            "id": "personalized",
            "purpose": "Give a personalized response using their name",
            "extraction_instructions": "Extract the user's name.",
            "response_instructions": "Use their name and ask how they're doing.",
            "required_context_keys": ["name"],
            "transitions": [{
                "target_state": "farewell",
                "description": "After user responds about their day"
            }]
        },
        "farewell": {
            "id": "farewell",
            "purpose": "Wish them well and say goodbye",
            "response_instructions": "Wish them well using their name and say goodbye.",
            "transitions": []
        }
    }
}

program = Program.from_fsm(greeting_fsm, model="gpt-4o-mini")

# First turn — auto-starts a conversation
result = program.invoke(message="Hi, I'm Alice.")
print(f"Bot: {result.value}")
conv_id = result.conversation_id

# Subsequent turns — pass the conversation id back
while True:
    user_input = input("You: ")
    if not user_input:
        break
    result = program.invoke(message=user_input, conversation_id=conv_id)
    print(f"Bot: {result.value}")
```

Or run the same FSM from the shell:

```bash
fsm-llm run greeting.json
```

## 5. Your first pipeline — `Program.from_term`

Stateless flows — `extract → reason → answer`, ReAct loops, debate, etc. — are written as λ-terms.

```python
from fsm_llm import Program, leaf, let_
from pydantic import BaseModel

class Topic(BaseModel):
    topic: str

term = let_(
    "topic", leaf(prompt="Extract the topic in one word: {q}", schema=Topic, input_var="q"),
    leaf(prompt="Write a one-paragraph article about {topic}.", input_var="topic"),
)

program = Program.from_term(term, model="gpt-4o-mini")
result = program.invoke(inputs={"q": "What is photosynthesis?"})

print(result.value)
assert result.oracle_calls == 2          # known ahead of time
```

## 6. Long-context with a hard cost gate — `Program.from_factory`

Chunk and recurse over a document too big for a single prompt:

```python
from fsm_llm import Program
from fsm_llm.stdlib.long_context import niah

program = Program.from_factory(niah, factory_kwargs={
    "question": "What animal is the protagonist?",
    "tau": 256, "k": 2,
})
result = program.invoke(inputs={"document": long_document})

print(result.value)
assert result.oracle_calls == result.plan.predicted_calls    # exact equality
```

The planner returns `(τ, k, depth, predicted_calls)` from the AST shape — so cost is a property of your program, not something you discover at runtime.

## 7. Add a Handler

Handlers add custom logic at 8 lifecycle points. Pass them to `Program.from_fsm` via `handlers=`:

```python
from fsm_llm import Program, HandlerBuilder, HandlerTiming

def detect_mood(context):
    response = str(context.get("_user_input", "")).lower()
    if any(w in response for w in ["good", "great", "wonderful"]):
        return {"mood": "positive"}
    if any(w in response for w in ["bad", "terrible", "awful"]):
        return {"mood": "negative", "offer_help": True}
    return {"mood": "neutral"}

mood_handler = (HandlerBuilder("MoodDetector")
                .at(HandlerTiming.POST_PROCESSING)
                .on_state("personalized")
                .do(detect_mood))

program = Program.from_fsm(greeting_fsm, model="gpt-4o-mini",
                           handlers=[mood_handler])
```

The eight timing points are: `START_CONVERSATION`, `PRE_PROCESSING`, `POST_PROCESSING`, `PRE_TRANSITION`, `POST_TRANSITION`, `CONTEXT_UPDATE`, `END_CONVERSATION`, `ERROR`. See [`docs/handlers.md`](handlers.md).

## 8. Useful Commands

```bash
fsm-llm run <target>                # FSM JSON path or pkg.mod:factory
fsm-llm explain <target>            # AST shape, leaf schemas, planner output
fsm-llm validate --fsm bot.json     # Validate FSM definition
fsm-llm visualize --fsm bot.json    # ASCII graph
fsm-llm monitor                     # Web dashboard at http://localhost:8420
```

## 9. Where to next

Three program shapes; pick the doc that matches yours:

- **Chatbot designer** — read [`docs/fsm_design.md`](fsm_design.md) for FSM patterns and anti-patterns.
- **Pipeline / agent author** — explore `examples/pipeline/` (λ-DSL twins of every agent pattern) and `fsm_llm.stdlib.agents`.
- **Long-context author** — see `examples/long_context/` for closed-form-cost demos.
- **Internals / theory** — read [`docs/lambda.md`](lambda.md) (architectural thesis) then [`docs/lambda_fsm_merge.md`](lambda_fsm_merge.md) (the merge contract).

## Migration from the legacy `API`

The `API` class still works in 0.5.x with no warnings — but `Program` is the unified entry going forward.

```python
# Legacy — still works
from fsm_llm import API
api = API.from_definition(greeting_fsm, model="gpt-4o-mini")
conv_id, hello = api.start_conversation()
reply = api.converse("hi", conv_id)

# Modern equivalent
from fsm_llm import Program
program = Program.from_fsm(greeting_fsm, model="gpt-4o-mini")
result = program.invoke(message="hi")
print(result.value, result.conversation_id)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No API key found` | Verify `echo $OPENAI_API_KEY` |
| `State not found` | Run `fsm-llm validate --fsm your_fsm.json` |
| `Context key missing` | Add a debug handler at `POST_PROCESSING` to print context keys |
| Cost drift in pipelines | Compare `result.oracle_calls` against `result.plan.predicted_calls` — non-equality means the program shape diverged from planner expectations |
