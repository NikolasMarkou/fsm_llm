# E-Commerce Customer Service with FSM Stacking

This example demonstrates advanced FSM stacking capabilities with a real-world product recommendation scenario.

## Purpose

This example shows how to:
- Start with a main customer service FSM
- Stack a specialized product recommendation FSM when needed
- Handle context inheritance and return between FSMs
- Use different merge strategies for context handover
- Preserve conversation history across FSM transitions

## FSM Structure

### Main Customer Service FSM
1. **greeting**: Welcome the customer and identify their needs
2. **general_help**: Handle general customer service inquiries
3. **product_inquiry**: Identify that customer needs product recommendations
4. **delegate_to_product_specialist**: Hand over to product specialist
5. **post_recommendation_followup**: Follow up after product recommendations
6. **resolution**: Confirm resolution and offer additional help
7. **farewell**: End the conversation

### Product Recommendation FSM (stacked)
1. **specialist_introduction**: Product specialist introduces themselves
2. **needs_assessment**: Gather comprehensive requirements
3. **recommendation_generation**: Present tailored recommendations
4. **recommendation_refinement**: Adjust based on feedback
5. **recommendation_finalization**: Summarize and provide next steps
6. **specialist_handoff**: Hand back to main customer service

## Key Concepts Demonstrated

- **`push_fsm()`**: Delegates to a specialized sub-FSM with context passing
- **`pop_fsm()`**: Returns to the parent FSM with merged context
- **`ContextMergeStrategy.UPDATE`**: Merges child FSM context into parent
- **`shared_context_keys`**: Keys that are shared between parent and child FSMs
- **`preserve_history`**: Maintains conversation history across FSM transitions

## How to Run

1. Make sure you have FSM-LLM installed:
```bash
pip install fsm-llm
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

3. Run the example:
```bash
python run.py
```

**Note:** FSMs in this example are defined inline as Python dicts (no JSON files). The example uses scripted/non-interactive conversations rather than user input.

Set `LLM_MODEL=ollama_chat/qwen3.5:4b` to use Ollama instead of OpenAI.

## Learning Points

This example demonstrates:
- How to build modular conversational applications using FSM stacking
- Context inheritance and handover between parent and child FSMs
- Using merge strategies to control how context flows back to the parent
- Building realistic customer service scenarios with delegation patterns
- Professional conversation flows with state transitions across FSM boundaries
