# Quick Start Tutorial

Welcome to FSM-LLM! This tutorial will have you building stateful conversations with LLMs in minutes.

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key (or another supported LLM provider)

## 1. Installation

```bash
pip install fsm-llm

# With extensions
pip install fsm-llm[reasoning]   # Structured reasoning engine
pip install fsm-llm[workflows]   # Workflow orchestration
pip install fsm-llm[agents]      # Agentic patterns (ReAct, HITL, meta-builder)
pip install fsm-llm[monitor]     # Real-time monitoring dashboard
pip install fsm-llm[all]         # Everything
```

Classification is built into the core package -- no extra install needed.

## 2. Set Your API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
# Or create a .env file:
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 3. Your First Bot

Create `hello_bot.py`:

```python
from fsm_llm import API

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

## 4. Add a Handler

Handlers add custom logic at 8 lifecycle points:

```python
from fsm_llm import API, HandlerTiming

api = API.from_definition(greeting_fsm, model="gpt-4o-mini")

def detect_mood(context):
    response = str(context.get("_user_input", "")).lower()
    positive = ["good", "great", "wonderful", "amazing"]
    negative = ["bad", "terrible", "awful", "stressed"]
    if any(w in response for w in positive):
        return {"mood": "positive"}
    elif any(w in response for w in negative):
        return {"mood": "negative", "offer_help": True}
    return {"mood": "neutral"}

api.register_handler(
    api.create_handler("MoodDetector")
        .at(HandlerTiming.POST_PROCESSING)
        .on_state("personalized")
        .do(detect_mood)
)
```

## 5. Next Steps

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
fsm-llm --fsm your_fsm.json           # Run any FSM interactively
fsm-llm-validate --fsm your_fsm.json  # Validate FSM structure
fsm-llm-visualize --fsm your_fsm.json # ASCII visualization
fsm-llm-monitor                       # Launch monitoring dashboard
```

### Common Patterns

**Information gathering** -- Use `required_context_keys` to stay in a state until data is collected.

**Branching logic** -- Multiple transitions with conditions route to different states.

**Error handling** -- Register a handler at `HandlerTiming.ERROR` for graceful recovery.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No API key found" | `echo $OPENAI_API_KEY` -- ensure it's set |
| "State not found" | Run `fsm-llm-validate --fsm your_fsm.json` |
| "Context key missing" | Add a debug handler at `POST_PROCESSING` to print context keys |
