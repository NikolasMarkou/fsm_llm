# Quick Start Tutorial - Get Running in 5 Minutes

Welcome to LLM-FSM! This tutorial will have you building stateful conversations with LLMs in just 5 minutes.

## Prerequisites

- Python 3.8 or higher
- An OpenAI API key (or another supported LLM provider)

## 1. Installation (30 seconds)

```bash
pip install llm-fsm
```

## 2. Set Your API Key (30 seconds)

```bash
# Option 1: Export as environment variable
export OPENAI_API_KEY="your-api-key-here"

# Option 2: Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 3. Your First Bot (2 minutes)

Create a file called `hello_bot.py`:

```python
from llm_fsm import API

# Define a simple greeting bot
greeting_fsm = {
    "name": "friendly_greeter",
    "initial_state": "welcome",
    "states": {
        "welcome": {
            "id": "welcome",
            "purpose": "Greet the user warmly and ask for their name",
            "transitions": [{
                "target_state": "personalized",
                "description": "After user provides their name"
            }]
        },
        "personalized": {
            "id": "personalized",
            "purpose": "Give a personalized response using their name and ask how they're doing",
            "required_context_keys": ["name"],
            "transitions": [{
                "target_state": "farewell",
                "description": "After user responds about their day"
            }]
        },
        "farewell": {
            "id": "farewell",
            "purpose": "Wish them well using their name and say goodbye",
            "transitions": []  # End state
        }
    }
}

# Create the bot
api = API.from_definition(greeting_fsm, model="gpt-4o-mini")

# Start conversation
conv_id, response = api.start_conversation()
print(f"Bot: {response}")

# Have a conversation
while not api.has_conversation_ended(conv_id):
    user_input = input("You: ")
    response = api.converse(user_input, conv_id)
    print(f"Bot: {response}")

# See what data was collected
print(f"\nCollected data: {api.get_data(conv_id)}")
```

Run it:
```bash
python hello_bot.py
```

**Expected Output:**
```
Bot: Hello there! Welcome! I'm so glad you're here. What's your name?
You: I'm Sarah
Bot: It's wonderful to meet you, Sarah! How has your day been so far?
You: Pretty good, just learning about LLM-FSM
Bot: That's fantastic, Sarah! I'm thrilled you're exploring LLM-FSM. I hope you find it useful for building amazing conversational experiences. Have a wonderful rest of your day!

Collected data: {'name': 'Sarah', '_extra': {'day_info': 'Pretty good, just learning about LLM-FSM'}}
```

## 4. Add Intelligence with Handlers (2 minutes)

Now let's make our bot smarter by adding a handler that responds to mood:

```python
from llm_fsm import API
from llm_fsm.handlers import HandlerTiming

# ... (same FSM definition as above) ...

api = API.from_definition(greeting_fsm, model="gpt-4o-mini")

# Add a mood detector
def detect_mood(context):
    """Detect user's mood from their response"""
    response = str(context.get("_user_input", "")).lower()
    
    positive_words = ["good", "great", "wonderful", "amazing", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "horrible", "stressed"]
    
    if any(word in response for word in positive_words):
        return {"mood": "positive", "mood_emoji": "😊"}
    elif any(word in response for word in negative_words):
        return {"mood": "negative", "mood_emoji": "😔", "offer_help": True}
    else:
        return {"mood": "neutral", "mood_emoji": "😐"}

# Register the handler
api.register_handler(
    api.create_handler("MoodDetector")
        .at(HandlerTiming.POST_PROCESSING)
        .on_state("personalized")
        .do(detect_mood)
)

# Run the enhanced bot
conv_id, response = api.start_conversation()
# ... conversation continues with mood-aware responses ...
```

## 5. Next Steps (30 seconds)

Congratulations! You've just built your first stateful conversation bot. Here's what to explore next:

### Try These Examples

1. **Form Bot** - Collect structured information:
   ```bash
   cd examples/basic/form_filling
   python main.py
   ```

2. **Quiz Bot** - Build an interactive quiz:
   ```bash
   cd examples/basic/quiz
   python main.py
   ```

3. **Customer Service** - Multi-branch conversations:
   ```bash
   cd examples/intermediate/customer_service
   python main.py
   ```

### Key Concepts to Master

1. **States** - Each conversation step
2. **Transitions** - Moving between states
3. **Context** - Remembering information
4. **Handlers** - Adding custom logic
5. **FSM Stacking** - Complex workflows

### Useful Commands

```bash
# Visualize any FSM
llm-fsm-visualize --fsm your_fsm.json

# Validate FSM structure
llm-fsm-validate --fsm your_fsm.json

# Run any FSM interactively
llm-fsm --fsm your_fsm.json
```

## Common Patterns

### Pattern 1: Information Gathering
```json
{
  "required_context_keys": ["email", "phone"],
  "purpose": "Collect contact information"
}
```

### Pattern 2: Branching Logic
```json
{
  "transitions": [
    {"target_state": "option_a", "description": "User chooses A"},
    {"target_state": "option_b", "description": "User chooses B"}
  ]
}
```

### Pattern 3: Error Handling
```python
api.register_handler(
    api.create_handler("ErrorHandler")
        .at(HandlerTiming.ERROR)
        .do(lambda ctx: {"error_handled": True})
)
```

## Troubleshooting

**Issue: "No API key found"**
```bash
# Make sure your API key is set
echo $OPENAI_API_KEY
```

**Issue: "State not found"**
```bash
# Validate your FSM
llm-fsm-validate --fsm your_fsm.json
```

**Issue: "Context key missing"**
```python
# Debug with logging
api.add_logging_handler()
```