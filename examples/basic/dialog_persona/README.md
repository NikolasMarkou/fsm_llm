# Simple Greeting Example

This example demonstrates a minimal implementation of a Finite State Machine (FSM) using LLM-FSM to conduct a basic greeting and farewell conversation.

## Purpose

This example shows how to:
- Create a simple conversational FSM with just a few states
- Handle basic user interactions
- Transition between states based on user input
- Use the simplified LLM_FSM API

## FSM Structure

This example implements a conversation with 3 states:
1. **greeting**: The initial state that welcomes the user and asks how they're doing
2. **conversation**: A middle state that responds to the user's feelings and offers assistance
3. **farewell**: A terminal state that ends the conversation with a goodbye message

## How to Run

1. Make sure you have LLM-FSM installed:
```bash
pip install llm-fsm
```

2. Set your OpenAI API key (or use the environment variable):
```bash
export OPENAI_API_KEY=your-api-key-here
```

3. Run the example:
```bash
python run.py
```

## Expected Output

```
System: Hello! I'm an AI assistant. How are you doing today?
You: I'm doing great, thanks for asking!
System: I'm glad to hear you're doing great! Is there anything I can help you with today?
You: No thanks, just saying hi
System: It was nice chatting with you! Have a wonderful day. Goodbye!
Conversation has ended.
```

## Learning Points

This example demonstrates:
- The minimal structure needed for an LLM-FSM definition
- How to define states and transitions
- How the LLM maintains conversational context within the FSM
- How terminal states work (the farewell state has no outgoing transitions)
- How to use the simplified LLM_FSM API for easier implementation