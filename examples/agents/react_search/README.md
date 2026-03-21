# ReAct Search Agent

Demonstrates the **ReAct (Reasoning + Acting)** pattern — the foundational agentic loop where an LLM reasons about a task, selects tools, observes results, and iterates until it can answer.

## What This Example Shows

- Registering multiple tools with `ToolRegistry`
- Creating a `ReactAgent` that auto-generates an FSM from the tool registry
- The think → act → observe → conclude loop
- How observations accumulate across iterations

## Tools

| Tool | Description |
|------|-------------|
| `search` | Simulates web search results |
| `calculate` | Evaluates math expressions |
| `lookup` | Looks up facts from a mock knowledge base |

## How to Run

```bash
# With OpenAI
export OPENAI_API_KEY=your-key-here
python examples/agents/react_search/run.py

# With Ollama
export LLM_MODEL=ollama_chat/qwen3.5:9b
python examples/agents/react_search/run.py
```

## Expected Output

```
Task: What is half the population of France?

Agent working on: What is half the population of France?
----------------------------------------

Answer: Half the population of France is approximately 34.2 million people.
Tools used: ['search', 'calculate']
Iterations: 4
```

## Learning Points

- The agent decides which tool to use based on the task
- Multiple tools can be chained (search first, then calculate)
- The agent terminates when it has enough information
- No FSM JSON is needed — the FSM is auto-generated from the tool registry
