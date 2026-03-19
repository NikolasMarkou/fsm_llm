# FSM-LLM Examples

Practical examples demonstrating FSM-LLM capabilities, organized by complexity.

## Getting Started

```bash
pip install fsm-llm
export OPENAI_API_KEY="your-key-here"
```

Each example contains:
- `run.py` — runnable Python script
- `*.json` — FSM definition file(s)
- `README.md` — detailed explanation

## Examples by Complexity

### Basic

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [simple_greeting](basic/simple_greeting/) | 3-state greeting bot | States, transitions, initial/terminal states |
| [form_filling](basic/form_filling/) | Data collection form | `required_context_keys`, data extraction |
| [story_time](basic/story_time/) | Interactive story narrator | Multi-turn conversation, context flow |

### Intermediate

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [book_recommendation](intermediate/book_recommendation/) | Book recommender | Conditional transitions, context-based routing |
| [product_recommendation](intermediate/product_recommendation/) | Product discovery assistant | Multi-step discovery, preference tracking |

### Advanced

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [yoga_instructions](advanced/yoga_instructions/) | Adaptive yoga instructor | User engagement tracking, non-linear FSM design |
| [e_commerce](advanced/e_commerce/) | E-commerce checkout flow | FSM stacking (`push_fsm`/`pop_fsm`), context merging |

### Classification

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [intent_routing](classification/intent_routing/) | Customer support intent classifier with handler routing | `Classifier`, `IntentRouter`, schema design |

## Suggested Learning Path

1. Start with **simple_greeting** to understand the basic FSM structure
2. Move to **form_filling** to learn about data extraction
3. Try **book_recommendation** for conditional transitions
4. Explore **e_commerce** for advanced FSM stacking
