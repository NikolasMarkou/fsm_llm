# FSM-LLM Examples

Practical examples demonstrating FSM-LLM capabilities, organized by complexity.

## Getting Started

```bash
pip install fsm-llm
export OPENAI_API_KEY="your-key-here"

# Or use a local Ollama model (no API key needed):
export LLM_MODEL="ollama_chat/qwen3.5:4b"
```

Each example contains:
- `run.py` — runnable Python script
- `*.json` — FSM definition file(s)

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
| [adaptive_quiz](intermediate/adaptive_quiz/) | Adaptive trivia quiz with handler-driven scoring | `HandlerBuilder` fluent API, `HandlerTiming`, runtime context mutation |

### Advanced

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [yoga_instructions](advanced/yoga_instructions/) | Adaptive yoga instructor | User engagement tracking, non-linear FSM design |
| [e_commerce](advanced/e_commerce/) | E-commerce checkout flow | FSM stacking (`push_fsm`/`pop_fsm`), context merging |
| [support_pipeline](advanced/support_pipeline/) | Full support pipeline with classification, stacking, and handlers | `Classifier` + `push_fsm`/`pop_fsm` + `HandlerBuilder`, multi-package integration |

### Classification

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [intent_routing](classification/intent_routing/) | Customer support intent classifier with handler routing | `Classifier`, `IntentRouter`, schema design |
| [smart_helpdesk](classification/smart_helpdesk/) | Classify intent then route to specialized FSM conversations | `Classifier` + `API.from_file()`, classification-driven FSM selection |

### Reasoning

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [math_tutor](reasoning/math_tutor/) | Math tutor combining conversational FSM with structured reasoning | `ReasoningEngine.solve_problem()`, injecting reasoning traces into FSM context |

### Workflows

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [order_processing](workflows/order_processing/) | Order pipeline with FSM conversation step inside a workflow | `WorkflowEngine`, `ConversationStep`, `api_step`, `condition_step`, async execution |

## Sub-Package Usage Matrix

| Example | Core FSM | Classification | Reasoning | Workflows | Handlers |
|---------|:--------:|:--------------:|:---------:|:---------:|:--------:|
| simple_greeting | x | | | | |
| form_filling | x | | | | |
| story_time | x | | | | |
| book_recommendation | x | | | | |
| product_recommendation | x | | | | |
| **adaptive_quiz** | x | | | | x |
| yoga_instructions | x | | | | |
| e_commerce | x | | | | |
| intent_routing | | x | | | |
| **smart_helpdesk** | x | x | | | |
| **math_tutor** | x | | x | | |
| **order_processing** | x | | | x | |
| **support_pipeline** | x | x | | | x |

## Suggested Learning Path

1. Start with **simple_greeting** to understand the basic FSM structure
2. Move to **form_filling** to learn about data extraction
3. Try **adaptive_quiz** to learn the handler system
4. Try **book_recommendation** for conditional transitions
5. Explore **smart_helpdesk** to combine classification with FSM conversations
6. Explore **math_tutor** to see reasoning engine integration
7. Explore **order_processing** to see workflow orchestration
8. Study **support_pipeline** for the full multi-package integration
